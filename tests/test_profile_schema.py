"""
Tests for the profile schema module (scripts/profile_schema.py).

Covers:
  - All six example YAML profiles parse without error
  - Required-field validation (missing model_file for llamacpp)
  - Hardware tier defaults
  - engine_args passthrough
  - model_params → engine CLI flag mapping
  - Invalid enum values are rejected
  - DGX Spark --enforce-eager injection
  - docker run spec structure
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

# Ensure scripts/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from profile_schema import (
    EngineType,
    HardwareTier,
    ModelParams,
    ProfileManifest,
    load_profile,
)
import pydantic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manifest(**overrides) -> ProfileManifest:
    """Create a minimal valid ProfileManifest with optional field overrides."""
    defaults = {
        "engine": "vllm",
        "hardware": "gpu",
        "model": "org/model-7b",
        "port": 8000,
    }
    defaults.update(overrides)
    return ProfileManifest(**defaults)


def make_llamacpp(**overrides) -> ProfileManifest:
    defaults = {
        "engine": "llamacpp",
        "hardware": "cpu",
        "model": "org/model-GGUF",
        "model_file": "model-q4_k_m.gguf",
        "port": 8002,
    }
    defaults.update(overrides)
    return ProfileManifest(**defaults)


# ---------------------------------------------------------------------------
# Example YAML profiles round-trip
# ---------------------------------------------------------------------------

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
EXAMPLE_PROFILES = list(PROFILES_DIR.glob("example-*.yaml"))


@pytest.mark.parametrize("profile_path", EXAMPLE_PROFILES, ids=lambda p: p.name)
def test_example_profiles_are_valid(profile_path: Path) -> None:
    """Every example-*.yaml in profiles/ must load and validate without error."""
    manifest = load_profile(str(profile_path))
    assert manifest.engine in EngineType
    assert manifest.hardware in HardwareTier
    assert manifest.port > 0
    assert manifest.model


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_missing_model_file_for_llamacpp_raises() -> None:
    with pytest.raises(pydantic.ValidationError, match="model_file"):
        ProfileManifest(engine="llamacpp", model="org/model-GGUF", port=8002)


def test_empty_model_string_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        ProfileManifest(engine="vllm", model="  ", port=8000)


def test_invalid_engine_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        ProfileManifest(engine="tensorrt", model="org/model", port=8000)


def test_invalid_hardware_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        ProfileManifest(engine="vllm", model="org/model", port=8000, hardware="tpu")


def test_port_out_of_range_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        ProfileManifest(engine="vllm", model="org/model", port=99999)


def test_gpu_memory_utilization_out_of_range_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        make_manifest(model_params={"gpu_memory_utilization": 1.5})


def test_extra_fields_in_manifest_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        ProfileManifest(engine="vllm", model="org/m", port=8000, unknown_field="x")


# ---------------------------------------------------------------------------
# Hardware tier defaults
# ---------------------------------------------------------------------------

def test_default_hardware_tier_is_gpu() -> None:
    m = make_manifest()
    assert m.hardware == HardwareTier.GPU


def test_cpu_tier() -> None:
    m = make_manifest(hardware="cpu")
    assert m.hardware == HardwareTier.CPU
    assert m.gpu_flags() == []


def test_gpu_tier_has_gpus_all() -> None:
    m = make_manifest(hardware="gpu")
    assert "--gpus" in m.gpu_flags()
    assert "all" in m.gpu_flags()


def test_dgx_spark_tier_has_gpus_all() -> None:
    m = make_manifest(hardware="dgx_spark")
    assert "--gpus" in m.gpu_flags()


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def test_vllm_cpu_image() -> None:
    m = make_manifest(hardware="cpu")
    assert "cpu" in m.engine_image()


def test_vllm_gpu_image() -> None:
    m = make_manifest(hardware="gpu")
    assert "vllm/vllm-openai" in m.engine_image()
    assert "cpu" not in m.engine_image()


def test_sglang_cpu_image() -> None:
    m = make_manifest(engine="sglang", hardware="cpu")
    assert "cpu" in m.engine_image()


def test_llamacpp_image() -> None:
    m = make_llamacpp()
    assert "llama.cpp" in m.engine_image()


# ---------------------------------------------------------------------------
# model_params → engine CLI flag mapping
# ---------------------------------------------------------------------------

def test_max_model_len_in_vllm_cmd() -> None:
    m = make_manifest(model_params={"max_model_len": 4096})
    cmd = m.build_engine_cmd()
    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_max_model_len_in_llamacpp_cmd() -> None:
    m = make_llamacpp(model_params={"max_model_len": 2048})
    cmd = m.build_engine_cmd()
    assert "-c" in cmd
    assert "2048" in cmd


def test_n_threads_in_llamacpp_cmd() -> None:
    m = make_llamacpp(model_params={"n_threads": 8})
    cmd = m.build_engine_cmd()
    assert "--threads" in cmd
    assert "8" in cmd


def test_n_parallel_in_llamacpp_cmd() -> None:
    m = make_llamacpp(model_params={"n_parallel": 4})
    cmd = m.build_engine_cmd()
    assert "--parallel" in cmd
    assert "4" in cmd


def test_n_gpu_layers_in_llamacpp_cmd() -> None:
    m = make_llamacpp(hardware="gpu", model_params={"n_gpu_layers": 20})
    cmd = m.build_engine_cmd()
    assert "--n-gpu-layers" in cmd
    assert "20" in cmd


def test_tensor_parallel_in_vllm_cmd() -> None:
    m = make_manifest(model_params={"tensor_parallel_size": 2})
    cmd = m.build_engine_cmd()
    assert "--tensor-parallel-size" in cmd
    assert "2" in cmd


def test_trust_remote_code_flag() -> None:
    m = make_manifest(model_params={"trust_remote_code": True})
    cmd = m.build_engine_cmd()
    assert "--trust-remote-code" in cmd


def test_trust_remote_code_not_present_when_false() -> None:
    m = make_manifest(model_params={"trust_remote_code": False})
    cmd = m.build_engine_cmd()
    assert "--trust-remote-code" not in cmd


def test_quantization_in_vllm_cmd() -> None:
    m = make_manifest(model_params={"quantization": "awq"})
    cmd = m.build_engine_cmd()
    assert "--quantization" in cmd
    assert "awq" in cmd


# ---------------------------------------------------------------------------
# engine_args passthrough
# ---------------------------------------------------------------------------

def test_engine_args_appended_last() -> None:
    m = make_manifest(engine_args=["--enable-prefix-caching", "--some-flag", "val"])
    cmd = m.build_engine_cmd()
    # Raw args should appear at the end
    idx = cmd.index("--enable-prefix-caching")
    assert idx == len(cmd) - 3


def test_engine_args_empty_by_default() -> None:
    m = make_manifest()
    # Engine args shouldn't appear in cmd if not provided
    assert "--enable-prefix-caching" not in m.build_engine_cmd()


# ---------------------------------------------------------------------------
# DGX Spark specific
# ---------------------------------------------------------------------------

def test_dgx_spark_injects_enforce_eager_for_vllm() -> None:
    m = make_manifest(hardware="dgx_spark")
    cmd = m.build_engine_cmd()
    assert "--enforce-eager" in cmd


def test_dgx_spark_no_double_enforce_eager() -> None:
    """If user already provides --enforce-eager in engine_args, it shouldn't duplicate."""
    m = make_manifest(hardware="dgx_spark", engine_args=["--enforce-eager"])
    cmd = m.build_engine_cmd()
    assert cmd.count("--enforce-eager") == 1


def test_dgx_spark_sglang_does_not_inject_enforce_eager() -> None:
    """--enforce-eager is vLLM-specific; SGLang on DGX Spark should not get it."""
    m = make_manifest(engine="sglang", hardware="dgx_spark")
    cmd = m.build_engine_cmd()
    assert "--enforce-eager" not in cmd


# ---------------------------------------------------------------------------
# Docker run spec
# ---------------------------------------------------------------------------

def test_docker_run_spec_keys() -> None:
    m = make_manifest()
    spec = m.to_docker_run_spec()
    required_keys = {
        "container_name", "image", "gpu_flags", "ports",
        "volumes", "env", "engine_cmd", "engine", "hardware",
        "model", "port", "cache_dir",
    }
    assert required_keys.issubset(spec.keys())


def test_docker_run_spec_llamacpp_has_models_dir() -> None:
    m = make_llamacpp()
    spec = m.to_docker_run_spec()
    assert "llamacpp_models_dir" in spec
    assert any("/models" in v for v in spec["volumes"])


def test_container_name_override() -> None:
    m = make_manifest(container_name="my-custom-container")
    assert m.resolved_container_name() == "my-custom-container"


def test_container_name_auto_generated() -> None:
    m = make_manifest(engine="vllm", port=9000)
    assert m.resolved_container_name() == "modeltainer-vllm-9000"
