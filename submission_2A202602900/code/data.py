"""data.py — Chuẩn bị và xử lý dữ liệu CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    path = Path(processed_dir)
    train_file = path / "train.npz"
    eval_file = path / "eval.npz"

    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file .npz tại {processed_dir}. Hãy chạy scripts/split_data.py trước."
        )

    train_data = np.load(train_file)
    eval_data = np.load(eval_file)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"].astype(np.int64)

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == N_FEATURES, (
        f"X_train_full shape không đúng: {X_train_full.shape}"
    )
    assert y_train_full.ndim == 1 and len(y_train_full) == len(X_train_full), (
        f"y_train_full shape không khớp: {y_train_full.shape}"
    )
    assert X_eval.ndim == 2 and X_eval.shape[1] == N_FEATURES, (
        f"X_eval shape không đúng: {X_eval.shape}"
    )
    assert y_eval.ndim == 1 and len(y_eval) == len(X_eval), (
        f"y_eval shape không khớp: {y_eval.shape}"
    )
    assert len(eval_row_id) == len(X_eval), (
        f"eval_row_id shape không khớp: {eval_row_id.shape}"
    )

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    cont = X_tr[:, :N_NUMERIC]
    mean = np.mean(cont, axis=0).astype(np.float32)
    std = np.std(cont, axis=0).astype(np.float32)
    eps = 1e-8
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_scaled = np.array(X, copy=True, dtype=np.float32)
    eps = 1e-8
    safe_std = np.where(std < eps, 1.0, std).astype(np.float32)
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / safe_std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr_std = apply_standardizer(X_tr, mean, std)
    X_val_std = apply_standardizer(X_val, mean, std)
    X_eval_std = apply_standardizer(X_eval, mean, std)

    torch_device = torch.device(device) if isinstance(device, str) else device

    X_tr_t = torch.as_tensor(X_tr_std, dtype=torch.float32, device=torch_device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64, device=torch_device)
    X_val_t = torch.as_tensor(X_val_std, dtype=torch.float32, device=torch_device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.int64, device=torch_device)
    X_eval_t = torch.as_tensor(X_eval_std, dtype=torch.float32, device=torch_device)
    y_eval_t = torch.as_tensor(y_eval, dtype=torch.int64, device=torch_device)

    # In thông tin các tập và baseline majority class
    bincount_val = np.bincount(y_val, minlength=N_CLASSES)
    majority_class = int(np.argmax(bincount_val))
    majority_acc = float(np.mean(y_val == majority_class))

    print(f"Kich thuoc tap du lieu:")
    print(f"  Train : X {X_tr_t.shape}, y {y_tr_t.shape} ({len(X_tr_t):,} mau)")
    print(f"  Val   : X {X_val_t.shape}, y {y_val_t.shape} ({len(X_val_t):,} mau)")
    print(f"  Eval  : X {X_eval_t.shape}, y {y_eval_t.shape} ({len(X_eval_t):,} mau)")
    print(f"Chien luoc luon doan lop da so (lop {majority_class}) tren val: accuracy = {majority_acc:.4f} (~0.4876)")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; giữ nguyên toàn bộ dữ liệu (không bỏ sót phần dư).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]
