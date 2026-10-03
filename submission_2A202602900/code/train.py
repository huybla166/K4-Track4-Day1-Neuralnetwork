"""train.py — Pipeline huấn luyện, đánh giá, dự đoán và thực nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    lname = loss_name.lower().strip()
    if lname == "ce":
        return F.cross_entropy(logits, y)
    elif lname == "mse":
        num_classes = logits.shape[1]
        y_one_hot = F.one_hot(y, num_classes=num_classes).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát không hỗ trợ: '{loss_name}'. Chọn 'ce' hoặc 'mse'.")


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    n = len(X)
    total_loss = 0.0
    all_preds = []

    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(xb)
        all_preds.append(logits.argmax(dim=-1))

    pred_tensor = torch.cat(all_preds, dim=0)
    avg_loss = total_loss / n
    acc = (pred_tensor == y).float().mean().item()

    # Tính Confusion Matrix
    y_cpu = y.cpu().numpy()
    pred_cpu = pred_tensor.cpu().numpy()
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_cpu, pred_cpu), 1)
    mf1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(mf1),
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    """
    seed = int(cfg.get("seed", 1))
    set_seed(seed)

    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Lệch số tham số: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"
    )

    precision = str(cfg.get("precision", "fp32")).lower().strip()
    use_amp = (precision == "fp16") and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_dtype = (
        torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else torch.float32)
    )

    opt_name = str(cfg.get("optimizer", "sgd_momentum"))
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    optimizer = build_optimizer(
        opt_name, model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    generator = torch.Generator(device=device).manual_seed(seed)

    loss_name = cfg.get("loss", "ce")
    # Bước 0: Đo loss val trước khi cập nhật bất kỳ bước nào
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = step0_res["loss"]

    batch_size = int(cfg.get("batch", 512))
    epochs = int(cfg.get("epochs", 20))
    clip_norm = cfg.get("clip_norm")
    if clip_norm is not None and str(clip_norm).lower() != "none":
        clip_norm = float(clip_norm)
    else:
        clip_norm = None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        batch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size,
                                      generator=generator, shuffle=True):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(precision != "fp32")):
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                batch_grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                batch_grad_norms.append(gn)
                optimizer.step()

        if diverged:
            print(f"Lần chạy {cfg.get('exp_id')} bị phân kỳ (NaN/inf loss) tại epoch {epoch}!")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # Đo train loss (chế độ eval, tính trên toàn bộ train) và val metrics
        train_eval = evaluate(model, data["X_tr"], data["y_tr"], loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)

        avg_gn = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_eval["loss"], 4))
        history["val_loss"].append(round(val_eval["loss"], 4))
        history["val_acc"].append(round(val_eval["acc"], 4))
        history["val_macro_f1"].append(round(val_eval["macro_f1"], 4))
        history["grad_norm"].append(round(avg_gn, 4))
        history["epoch_time_s"].append(round(epoch_time, 2))

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Lấy metrics tại best_epoch
    best_idx = best_epoch - 1 if (history["epoch"] and best_epoch <= len(history["epoch"])) else -1
    summary = {
        "step0_loss": round(float(step0_loss), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "best_epoch": int(best_epoch),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
        "val_acc": history["val_acc"][best_idx] if best_idx >= 0 else 0.0,
        "val_macro_f1": history["val_macro_f1"][best_idx] if best_idx >= 0 else 0.0,
        "time_per_epoch_s": round(float(np.mean(history["epoch_time_s"])), 2) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": round(float(peak_mem_MB), 1),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)})
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = 0.0  # Không bao giờ dùng dropout khi dự đoán eval
    model = MLP(hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)

