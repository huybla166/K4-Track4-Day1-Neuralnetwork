"""optimizer.py — Quản lý bộ tối ưu hoá, scheduler và gradient clipping.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8) -> torch.optim.Optimizer:
    """Trả về một torch.optim.Optimizer.

    Các bước:
      1. kiểm tra name nằm trong OPTIMIZERS, nếu không raise ValueError
      2. "sgd"          -> torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
         "sgd_momentum" -> torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
         "adam"         -> torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
         "adamw"        -> torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    Chú ý: weight_decay của Adam (L2 trộn vào gradient) khác weight_decay của AdamW (suy giảm tách riêng).
    """
    opt_name = name.lower().strip()
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không nằm trong {OPTIMIZERS}")

    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Không hỗ trợ: {opt_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (slide có ví dụ CosineAnnealingLR).

    Trả về None nếu name là None. Nếu bạn dùng scheduler ở một thí nghiệm, hãy ghi vào bảng (cột notes).
    """
    if name is None:
        return None
    sched_name = name.lower().strip()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    elif sched_name == "step":
        step_size = kwargs.get("step_size", max(1, total_steps // 3))
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Scheduler không hỗ trợ: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None: tính chuẩn toàn cục mà không cắt (ví dụ clip_grad_norm_ với max_norm=inf)
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    Giá trị trả về chính là `grad_norm` bạn phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    # Lấy danh sách tham số có gradient
    param_list = [p for p in params if p.grad is not None]
    if not param_list:
        return 0.0

    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float(max_norm))

    return float(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)

