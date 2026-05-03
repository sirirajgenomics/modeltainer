#!/usr/bin/env python3
"""
ModelTainer Profile Schema
==========================
Validates and resolves YAML profile manifests into structured Docker arguments.

Usage (standalone validation):
    python scripts/profile_schema.py profiles/my-profile.yaml

Usage (emit resolved JSON for run_profile.sh):
    python scripts/profile_schema.py --json profiles/my-profile.yaml
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HardwareTier(str, Enum):
    """Supported hardware execution tiers."""

    CPU = "cpu"
    """CPU-only machine — no GPU available. Engine images must support CPU inference."""

    GPU = "gpu"
    """Standard CPU + NVIDIA/AMD GPU machine. Uses --gpus all."""

    DGX_SPARK = "dgx_spark"
    """NVIDIA DGX Spark (GB10 Grace Blackwell). ARM aarch64 + unified 128 GB memory.
    Uses --gpus all with Blackwell-specific flags (e.g., --enforce-eager for vLLM)."""


class EngineType(str, Enum):
    """Supported inference engine backends."""

    VLLM = "vllm"
    """vLLM — high-throughput GPU inference (also supports CPU mode via --device cpu)."""

    SGLANG = "sglang"
    """SGLang — optimised for structured generation and high throughput."""

    LLAMACPP = "llamacpp"
    """llama.cpp — GGUF-based inference, excellent on CPU and low-VRAM GPU."""


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ModelParams(BaseModel):
    """Structured model tunables that map 1:1 to engine CLI flags."""

    # Context / memory
    max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum sequence length (context window). Maps to --max-model-len (vLLM/SGLang) or -c (llama.cpp).",
        ge=1,
    )
    kv_cache_dtype: Optional[str] = Field(
        default=None,
        description="KV cache data type. e.g. 'auto', 'fp8', 'fp8_e5m2'. vLLM/SGLang only.",
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=None,
        description="Fraction of GPU memory to use for the KV cache (0.0–1.0). vLLM/SGLang only.",
        ge=0.0,
        le=1.0,
    )

    # Precision / quantisation
    quantization: Optional[str] = Field(
        default=None,
        description="Quantisation method. e.g. 'awq', 'gptq', 'fp8'. vLLM/SGLang only.",
    )
    dtype: Optional[str] = Field(
        default=None,
        description="Model weights dtype. e.g. 'auto', 'half', 'float16', 'bfloat16'.",
    )

    # Parallelism
    tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="Number of GPUs for tensor parallelism. vLLM/SGLang only.",
        ge=1,
    )
    pipeline_parallel_size: Optional[int] = Field(
        default=None,
        description="Number of pipeline stages. vLLM only.",
        ge=1,
    )

    # CPU-specific (llama.cpp)
    n_threads: Optional[int] = Field(
        default=None,
        description="Number of CPU threads. Maps to --threads (-t) in llama.cpp.",
        ge=1,
    )
    n_parallel: Optional[int] = Field(
        default=None,
        description="Number of parallel request slots. Maps to --parallel (-np) in llama.cpp.",
        ge=1,
    )
    n_gpu_layers: Optional[int] = Field(
        default=None,
        description="Number of model layers to offload to GPU. Maps to --n-gpu-layers (-ngl) in llama.cpp.",
        ge=-1,
    )

    # Misc
    trust_remote_code: bool = Field(
        default=False,
        description="Allow custom model code from Hugging Face. Maps to --trust-remote-code.",
    )
    served_model_name: Optional[str] = Field(
        default=None,
        description="Override the model name advertised by the engine OpenAI API. Maps to --served-model-name.",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------

class ProfileManifest(BaseModel):
    """
    A ModelTainer profile manifest — the single file that describes exactly
    how to launch one inference engine backend.
    """

    # --- Required ---
    engine: EngineType = Field(description="Inference engine to use.")
    model: str = Field(description="Hugging Face repo ID (all engines) or local path.")
    port: int = Field(default=8000, description="Host port to expose.", ge=1, le=65535)

    # --- Hardware ---
    hardware: HardwareTier = Field(
        default=HardwareTier.GPU,
        description="Hardware tier — controls GPU attachment and image variant selection.",
    )

    # --- llama.cpp specific ---
    model_file: Optional[str] = Field(
        default=None,
        description="GGUF filename within the HF repository. Required for llamacpp engine.",
    )

    # --- Tuning ---
    model_params: ModelParams = Field(
        default_factory=ModelParams,
        description="Structured model parameters mapped to engine CLI flags.",
    )

    # --- Raw passthrough ---
    engine_args: List[str] = Field(
        default_factory=list,
        description="Additional raw arguments appended verbatim to the engine command.",
    )

    # --- Infrastructure ---
    host_cache_dir: Optional[str] = Field(
        default=None,
        description="Host directory for model cache. Defaults to ~/.cache/modeltainer.",
    )
    container_name: Optional[str] = Field(
        default=None,
        description="Override the generated container name (modeltainer-{engine}-{port}).",
    )
    hf_token_env: Optional[str] = Field(
        default=None,
        description="Name of the env var holding the Hugging Face token, e.g. HF_TOKEN.",
    )

    model_config = {"extra": "forbid"}

    # --- Validators ---

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be empty")
        return v

    @model_validator(mode="after")
    def llamacpp_requires_model_file(self) -> "ProfileManifest":
        if self.engine == EngineType.LLAMACPP and not self.model_file:
            raise ValueError(
                "model_file is required for the llamacpp engine "
                "(e.g. model_file: my-model-q4_k_m.gguf)"
            )
        return self

    @model_validator(mode="after")
    def cpu_tier_no_gpu_params(self) -> "ProfileManifest":
        """Warn-level: GPU-specific params on a CPU-only profile are ignored."""
        # Not an error — user may be writing a generic template — just pass through.
        return self

    # --- Resolution helpers ---

    def resolved_container_name(self) -> str:
        return self.container_name or f"modeltainer-{self.engine.value}-{self.port}"

    def resolved_cache_dir(self) -> str:
        import os
        return self.host_cache_dir or os.path.expanduser("~/.cache/modeltainer")

    def gpu_flags(self) -> List[str]:
        """Return Docker GPU flags appropriate for the hardware tier."""
        if self.hardware == HardwareTier.CPU:
            return []
        # GPU and DGX_SPARK both need GPU access
        return ["--gpus", "all"]

    def engine_image(self) -> str:
        """Select the official Docker image for this engine + hardware combination."""
        if self.engine == EngineType.VLLM:
            if self.hardware == HardwareTier.CPU:
                # vLLM ships a CPU-only image
                return "vllm/vllm-openai:latest-cpu"
            # GPU and DGX Spark both use the standard GPU image
            # DGX Spark (aarch64/Blackwell) uses the same tag; NVIDIA ships multi-arch manifests
            return "vllm/vllm-openai:latest"

        if self.engine == EngineType.SGLANG:
            if self.hardware == HardwareTier.CPU:
                # SGLang CPU mode via GGUF backend
                return "lmsysorg/sglang:latest-cpu"
            return "lmsysorg/sglang:latest"

        if self.engine == EngineType.LLAMACPP:
            # llama.cpp server image — same image handles CPU/GPU via -ngl flag
            return "ghcr.io/ggerganov/llama.cpp:server"

        raise ValueError(f"Unknown engine: {self.engine}")  # pragma: no cover

    def build_engine_cmd(self) -> List[str]:
        """Build the engine command list from structured model_params + raw engine_args."""
        mp = self.model_params
        cmd: List[str] = []

        if self.engine in (EngineType.VLLM, EngineType.SGLANG):
            cmd += self._vllm_sglang_cmd()
        elif self.engine == EngineType.LLAMACPP:
            cmd += self._llamacpp_cmd()

        # Append raw passthrough args last
        cmd += self.engine_args
        return cmd

    def _vllm_sglang_cmd(self) -> List[str]:
        mp = self.model_params
        cmd: List[str] = []

        if self.engine == EngineType.VLLM:
            cmd += ["--model", self.model, "--port", str(self.port), "--host", "0.0.0.0"]
        else:  # sglang
            cmd += ["--model-path", self.model, "--port", str(self.port), "--host", "0.0.0.0"]

        # DGX Spark: enforce eager to avoid CUDA graph issues on Grace Blackwell
        if self.hardware == HardwareTier.DGX_SPARK and self.engine == EngineType.VLLM:
            if "--enforce-eager" not in self.engine_args:
                cmd.append("--enforce-eager")

        # CPU mode for vLLM
        if self.hardware == HardwareTier.CPU and self.engine == EngineType.VLLM:
            cmd += ["--device", "cpu"]

        if mp.max_model_len is not None:
            cmd += ["--max-model-len", str(mp.max_model_len)]
        if mp.kv_cache_dtype is not None:
            cmd += ["--kv-cache-dtype", mp.kv_cache_dtype]
        if mp.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(mp.gpu_memory_utilization)]
        if mp.quantization is not None:
            cmd += ["--quantization", mp.quantization]
        if mp.dtype is not None:
            cmd += ["--dtype", mp.dtype]
        if mp.tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(mp.tensor_parallel_size)]
        if mp.pipeline_parallel_size is not None and self.engine == EngineType.VLLM:
            cmd += ["--pipeline-parallel-size", str(mp.pipeline_parallel_size)]
        if mp.trust_remote_code:
            cmd.append("--trust-remote-code")
        if mp.served_model_name is not None:
            cmd += ["--served-model-name", mp.served_model_name]

        return cmd

    def _llamacpp_cmd(self) -> List[str]:
        mp = self.model_params
        cmd: List[str] = [
            "-m", f"/models/{self.model_file}",
            "--port", str(self.port),
            "--host", "0.0.0.0",
        ]

        if mp.max_model_len is not None:
            cmd += ["-c", str(mp.max_model_len)]
        if mp.n_threads is not None:
            cmd += ["--threads", str(mp.n_threads)]
        if mp.n_parallel is not None:
            cmd += ["--parallel", str(mp.n_parallel)]
        if mp.n_gpu_layers is not None:
            cmd += ["--n-gpu-layers", str(mp.n_gpu_layers)]

        return cmd

    def to_docker_run_spec(self) -> Dict[str, Any]:
        """
        Emit a structured JSON spec that run_profile.sh can consume to
        build the final `docker run` command without any Bash string parsing.
        """
        import os
        env_vars: Dict[str, str] = {}
        if self.hf_token_env:
            token = os.environ.get(self.hf_token_env, "")
            if token:
                env_vars["HUGGING_FACE_HUB_TOKEN"] = token
                env_vars["HF_TOKEN"] = token

        cache_dir = self.resolved_cache_dir()

        spec: Dict[str, Any] = {
            "container_name": self.resolved_container_name(),
            "image": self.engine_image(),
            "gpu_flags": self.gpu_flags(),
            "ports": [f"{self.port}:{self.port}"],
            "volumes": [],
            "env": env_vars,
            "engine_cmd": [],
            "engine": self.engine.value,
            "hardware": self.hardware.value,
            "model": self.model,
            "port": self.port,
            "cache_dir": cache_dir,
        }

        if self.engine == EngineType.LLAMACPP:
            # llama.cpp uses a dedicated models dir; cache is only for HF downloads
            models_dir = os.path.expanduser("~/.cache/modeltainer_llamacpp")
            spec["volumes"].append(f"{models_dir}:/models")
            spec["llamacpp_models_dir"] = models_dir
            spec["model_file"] = self.model_file
        else:
            spec["volumes"].append(f"{cache_dir}:/root/.cache/huggingface")

        spec["engine_cmd"] = self.build_engine_cmd()
        return spec


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def load_profile(path: str) -> ProfileManifest:
    """Load and validate a profile YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProfileManifest(**data)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate or resolve a ModelTainer profile YAML manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("profile", help="Path to the .yaml profile file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the resolved docker-run spec as JSON (consumed by run_profile.sh).",
    )
    args = parser.parse_args()

    try:
        manifest = load_profile(args.profile)
    except Exception as exc:
        print(f"ERROR: Profile validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(manifest.to_docker_run_spec(), indent=2))
    else:
        print(f"✅  Profile '{args.profile}' is valid.")
        print(f"    engine   : {manifest.engine.value}")
        print(f"    hardware : {manifest.hardware.value}")
        print(f"    model    : {manifest.model}")
        print(f"    port     : {manifest.port}")
        print(f"    image    : {manifest.engine_image()}")
        print(f"    cmd      : {' '.join(manifest.build_engine_cmd())}")


if __name__ == "__main__":
    main()
