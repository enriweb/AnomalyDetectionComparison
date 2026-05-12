"""Inference-efficiency metrics shared across model scripts.

Public API:
    count_params(model)                          -> int
    count_flops(model, input_shape)              -> int | None
    gpu_throughput(model, input_shape, ...)      -> float (img/s)
    cpu_latency(model, input_shape, ...)         -> float (ms/img)
    peak_gpu_mem(model, input_shape)             -> int (bytes)
    measure_all(model, input_shape, device, ...) -> dict
"""

from __future__ import annotations

import copy
import gc
import time
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn

try:
    from thop import profile as _thop_profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis as _FvFlop
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False


def _default_forward(model: nn.Module, x: torch.Tensor):
    return model(x)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_flops(
    model: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    device: str | torch.device = "cpu",
) -> Optional[int]:
    """FLOPs for single forward. Returns None if no profiler works."""
    model = model.eval()
    x = torch.randn(*input_shape, device=device)
    if _HAS_THOP:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flops, _ = _thop_profile(model, inputs=(x,), verbose=False)
            return int(flops)
        except Exception:
            pass
    if _HAS_FVCORE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return int(_FvFlop(model, x).total())
        except Exception:
            pass
    return None


@torch.no_grad()
def gpu_throughput(
    model: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    batch_size: int = 32,
    n_warmup: int = 10,
    n_iter: int = 50,
    forward_fn: Optional[Callable] = None,
) -> Optional[float]:
    """Images per second on CUDA. Returns None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    fwd = forward_fn or _default_forward
    model = model.cuda().eval()
    _, c, h, w = input_shape
    x = torch.randn(batch_size, c, h, w, device="cuda")
    for _ in range(n_warmup):
        fwd(model, x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fwd(model, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (batch_size * n_iter) / elapsed


@torch.no_grad()
def cpu_latency(
    model: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    n_warmup: int = 5,
    n_iter: int = 20,
    num_threads: int = 1,
    forward_fn: Optional[Callable] = None,
) -> float:
    """Mean ms per single-image forward on CPU."""
    fwd = forward_fn or _default_forward
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    cpu_model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(*input_shape)
    try:
        for _ in range(n_warmup):
            fwd(cpu_model, x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fwd(cpu_model, x)
        elapsed = time.perf_counter() - t0
    finally:
        torch.set_num_threads(prev_threads)
        del cpu_model
        gc.collect()
    return (elapsed / n_iter) * 1000.0


@torch.no_grad()
def peak_gpu_mem(
    model: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    forward_fn: Optional[Callable] = None,
) -> Optional[int]:
    """Peak GPU memory in bytes for a single forward. None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    fwd = forward_fn or _default_forward
    model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randn(*input_shape, device="cuda")
    fwd(model, x)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_all(
    model: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    device: str | torch.device = "cuda",
    forward_fn: Optional[Callable] = None,
    skip_cpu: bool = False,
    skip_flops: bool = False,
    gpu_batch: int = 32,
) -> dict:
    """Return all efficiency metrics in a single dict.

    Notes:
        - Caller passes the nn.Module to benchmark (e.g. a feature extractor,
          or the full model). Wrap with `forward_fn` if `model(x)` has an
          incompatible signature.
        - FLOPs computed on CPU copy to avoid disturbing GPU state.
    """
    out: dict = {}
    out["params"] = count_params(model)
    out["params_trainable"] = count_params(model, trainable_only=True)

    if not skip_flops:
        try:
            cpu_clone = copy.deepcopy(model).cpu()
            out["flops"] = count_flops(cpu_clone, input_shape=input_shape, device="cpu")
            del cpu_clone
            gc.collect()
        except Exception as e:
            out["flops"] = None
            out["flops_error"] = str(e)
    else:
        out["flops"] = None

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            out["gpu_throughput"] = gpu_throughput(
                model, input_shape=input_shape, batch_size=gpu_batch,
                forward_fn=forward_fn,
            )
        except Exception as e:
            out["gpu_throughput"] = None
            out["gpu_throughput_error"] = str(e)
        try:
            out["peak_gpu_mem"] = peak_gpu_mem(
                model, input_shape=input_shape, forward_fn=forward_fn,
            )
        except Exception as e:
            out["peak_gpu_mem"] = None
            out["peak_gpu_mem_error"] = str(e)
    else:
        out["gpu_throughput"] = None
        out["peak_gpu_mem"] = None

    if not skip_cpu:
        try:
            out["cpu_latency_ms"] = cpu_latency(
                model, input_shape=input_shape, forward_fn=forward_fn,
            )
        except Exception as e:
            out["cpu_latency_ms"] = None
            out["cpu_latency_error"] = str(e)
    else:
        out["cpu_latency_ms"] = None

    return out
