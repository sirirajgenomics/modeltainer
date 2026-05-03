import os
import time

import pytest
from pydantic import ValidationError

from api.registry import Registry


def test_registry_reload(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("models:\n  a:\n    backend_url: http://one\n")
    reg = Registry(str(cfg))
    assert "a" in reg.models
    time.sleep(0.1)
    cfg.write_text("models:\n  b:\n    backend_url: http://two\n")
    reg.maybe_reload()
    assert "b" in reg.models
    assert "a" not in reg.models


def test_registry_invalid(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("models:\n  a:\n    backend_url: http://one\n")
    reg = Registry(str(cfg))
    time.sleep(0.1)
    cfg.write_text("models:\n  bad:\n    foo: bar\n")
    with pytest.raises(ValidationError):
        reg.maybe_reload()
    assert "a" in reg.models


def test_env_applied(tmp_path, monkeypatch):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("models:\n  a:\n    backend_url: http://one\n    env:\n      TEST_VAR: value\n")
    monkeypatch.delenv("TEST_VAR", raising=False)
    Registry(str(cfg))
    assert os.environ["TEST_VAR"] == "value"
    monkeypatch.delenv("TEST_VAR", raising=False)


def test_sglang_backend_is_valid(tmp_path):
    """sglang must be a valid backend value in models.yaml (it was missing from the enum)."""
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "models:\n"
        "  my-sglang-model:\n"
        "    backend: sglang\n"
        "    backend_url: http://localhost:8001\n"
    )
    reg = Registry(str(cfg))
    assert "my-sglang-model" in reg.models
    from api.registry import Backend
    assert reg.models["my-sglang-model"].backend == Backend.SGLANG
