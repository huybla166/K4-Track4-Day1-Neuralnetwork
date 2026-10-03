"""plots.py — Vẽ biểu đồ đường cong huấn luyện và so sánh thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})

    epochs = hist["epoch"]
    best_epoch = summary.get("best_epoch", epochs[-1] if epochs else 1)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Cấu hình chuỗi tiêu đề
    title_str = (
        f"Experiment: {cfg.get('exp_id', 'exp')} | Opt: {cfg.get('optimizer')} (lr={cfg.get('lr')}) | "
        f"Batch: {cfg.get('batch')} | Epochs: {cfg.get('epochs')} | Hidden: {cfg.get('hidden')}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=1.02)

    # Ô 1: Train loss & Val loss
    ax1 = axes[0]
    ax1.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="tab:blue", lw=2)
    ax1.plot(epochs, hist["val_loss"], label="Val Loss", color="tab:orange", lw=2)
    ax1.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Loss vs. Epoch", fontsize=11)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # Ô 2: Val accuracy & Val macro-F1
    ax2 = axes[1]
    ax2.plot(epochs, hist["val_acc"], label="Val Accuracy", color="tab:green", lw=2)
    if "val_macro_f1" in hist:
        ax2.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="tab:purple", lw=2)
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax2.set_title("Validation Metrics vs. Epoch", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=9)

    # Ô 3: Gradient norm
    ax3 = axes[2]
    ax3.plot(epochs, hist["grad_norm"], label="Grad Norm (L2, pre-clip)", color="tab:red", lw=2)
    ax3.set_title("Average Gradient Norm vs. Epoch", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Grad Norm")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        epochs = hist["epoch"]
        if metric in hist:
            values = hist[metric]
            label = f"{exp_id} ({cfg.get('optimizer')}, lr={cfg.get('lr')})"
            ax.plot(epochs, values, label=label, lw=2)

    ax.set_title(title or f"Comparison: {metric}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

