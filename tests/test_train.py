"""Hardware-Erkennung: welche Precision auf welcher GPU.

Der Test, der eine echte Colab-T4 gefunden hat: ``is_bf16_supported()``
meldet dort ``True``, obwohl Turing kein bf16 kann. Deshalb entscheidet die
Compute Capability.
"""

from __future__ import annotations

import types

import pytest

from lfm3ha.train import check_peft_env, pick_dtype


def fake_torch(capability, cuda=True, mps=False, bf16_flag=True):
    """Minimales torch-Attrappenmodell für ``pick_dtype``."""

    torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            get_device_capability=lambda _i=0: capability,
            is_bf16_supported=lambda: bf16_flag,
        ),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
    )
    return torch


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((7, 5), "fp16"),   # T4 – bf16 würde emuliert und NaN liefern
        ((8, 0), "bf16"),   # A100
        ((8, 9), "bf16"),   # L4 / L40S
        ((9, 0), "bf16"),   # H100
        ((6, 1), "fp16"),   # P100
    ],
)
def test_pick_dtype_folgt_der_compute_capability(capability, expected):
    assert pick_dtype(fake_torch(capability)) == expected


def test_pick_dtype_ignoriert_das_bf16_flag():
    """Der eigentliche Fund: Flag True, Karte kann es trotzdem nicht."""
    torch = fake_torch((7, 5), bf16_flag=True)
    assert pick_dtype(torch) == "fp16"


def test_pick_dtype_ohne_cuda():
    assert pick_dtype(fake_torch((7, 5), cuda=False)) == "fp32"
    assert pick_dtype(fake_torch((7, 5), cuda=False, mps=True)) == "fp16"


def test_check_peft_env_ohne_torchao(monkeypatch):
    """Nicht installiert ist der Normalfall und darf nichts tun."""

    def boom(_name: str) -> str:
        from importlib.metadata import PackageNotFoundError

        raise PackageNotFoundError("torchao")

    monkeypatch.setattr("importlib.metadata.version", boom)
    assert check_peft_env() is None


def test_check_peft_env_meldet_alte_torchao_version(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _n: "0.10.0")
    with pytest.raises(SystemExit, match="torchao 0.10.0"):
        check_peft_env()


def test_check_peft_env_akzeptiert_neue_version(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _n: "0.18.0")
    assert check_peft_env() is None
