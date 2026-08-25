import os
import sys
import json
import csv
import random
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ========================= USER SETTINGS =========================
#H5_PATH = "/home/YTX/GLdata/APR_6000_v2.h5"
H5_PATH = "/home/YTX/GLdata/MERGE_E45nN5.h5"
SPLIT_CSV_PATH = "/home/YTX/SeismoDual/MERGE_E45nN5_sb.csv"   # CSV must contain: trace_name, split
TRACE_COL = "trace_name"
SPLIT_COL = "split"

MODEL_DEF_PATH = "./model/"
PRETRAINED_MODEL_PATH = "./trainCWALFB/model_100hz_finetune_best.pth"
#PRETRAINED_MODEL_PATH = "./model/model_100Hz.pth"
SAVE_DIR = "./finetune_CWALFB_s45n5"
method='LFB'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Window extraction from 60 s / 6000 samples HDF5 traces
# Options: "first30", "middle30", "custom", "p_center", "p_random"
WINDOW_MODE = "p_random"
CUSTOM_START_SAMPLE = 0

# Data filtering
ONLY_COMPLETE = True
RECEIVER_TYPE = None   # "HH", "EH", or None

# Label generation
LABEL_HALF_WIDTH = 40
BACKGROUND_CLASS = 2   # P=0, S=1, BG=2

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
EARLY_STOPPING_PATIENCE = 7
MIN_DELTA = 0.0

# Loss weighting
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 1.0, 0.2]

# Save
BEST_MODEL_NAME = "model_100hz_finetune_best.pth"
LAST_MODEL_NAME = "model_100hz_finetune_last.pth"
HISTORY_JSON_NAME = "train_history.json"
# ================================================================


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_csv(csv_path, trace_col="trace_name", split_col="split"):
    split_map = {}
    valid_splits = {"train", "dev", "test"}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if trace_col not in reader.fieldnames:
            raise KeyError(f"CSV missing trace column: {trace_col}")
        if split_col not in reader.fieldnames:
            raise KeyError(f"CSV missing split column: {split_col}")

        for row in reader:
            trace_name = str(row[trace_col]).strip()
            split_name = str(row[split_col]).strip().lower()

            if split_name not in valid_splits:
                raise ValueError(
                    f"Invalid split '{split_name}' for trace '{trace_name}'. "
                    f"Allowed: {sorted(valid_splits)}"
                )

            split_map[trace_name] = split_name

    if len(split_map) == 0:
        raise RuntimeError("Split CSV is empty.")

    return split_map


def zscore_safe(x):
    x = np.asarray(x, dtype=np.float32)
    std = np.std(x)
    if std == 0 or np.isnan(std):
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def build_seg_label(length, p_idx, s_idx, half_width=20):
    """
    Class mapping:
      0 = P
      1 = S
      2 = background
    """
    y = np.full(length, BACKGROUND_CLASS, dtype=np.int64)

    if 0 <= p_idx < length:
        l = max(0, p_idx - half_width)
        r = min(length, p_idx + half_width + 1)
        y[l:r] = 0

    if 0 <= s_idx < length:
        l = max(0, s_idx - half_width)
        r = min(length, s_idx + half_width + 1)
        y[l:r] = 1

    return y


def choose_window_start_random_ps(attrs, npts):#, min_p=300, max_p=900):
    if npts < 3000:
        return None

    p = attrs.get("p_arrival_sample", None)
    s = attrs.get("s_arrival_sample", None)
    
    if p is None or s is None:
        return None
    
    try:
        p = int(p)
        s = int(s)
    except Exception:
        return None
    if s-p<=2800:

        target_p = np.random.randint(10, 2900-s+p)
        s0 = p - target_p
        s0 = max(0, min(s0, npts - 3000))
    
        p_in = p - s0
        s_in = s - s0
        if not (0 <= p_in < 3000 and 0 <= s_in < 3000):
            return None
    
    return int(s0)


def choose_window_start(attrs, npts, window_mode="first30", start_sample=0):
    if npts < 3000:
        return None

    if window_mode == "first30":
        s0 = 0
    elif window_mode == "middle30":
        s0 = max(0, (npts - 3000) // 2)
    elif window_mode == "custom":
        s0 = int(start_sample)
    elif window_mode == "p_center":
        p = attrs.get("p_arrival_sample", None)
        if p is None:
            return 100
        try:
            p = int(p)
        except Exception:
            return None
        s0 = p - 500
    elif window_mode == "p_random":
        p = attrs.get("p_arrival_sample", None)
        if p is None:
            return 100
        try:
            p = int(p)
        except Exception:
            return None

        s0 = choose_window_start_random_ps(attrs, npts)
        if s0 is None:
            #target_p = np.random.randint(300, 901)
            s0 = 100#p - target_p
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")

    s0 = max(0, min(int(s0), npts - 3000))
    return int(s0)


def load_h5_3c_100hz_training_data(
    h5_path,
    split_map,
    use_split,
    window_mode="first30",
    start_sample=0,
    only_complete=True,
    receiver_type=None,
    label_half_width=20,
):
    if use_split not in {"train", "dev", "test"}:
        raise ValueError(f"use_split must be train/dev/test, got: {use_split}")

    X_list = []
    Y_list = []
    meta_list = []

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"'data' group not found in HDF5: {h5_path}")

        g = f["data"]

        for trace_name in g.keys():
            trace_split = split_map.get(trace_name, None)
            if trace_split != use_split:
                continue

            ds = g[trace_name]
            attrs = ds.attrs
            data = ds[()]

            '''
            sr = float(attrs.get("sampling_rate", -1))
            if abs(sr - 100.0) > 1e-6:
                continue

            if data.ndim != 2 or data.shape[1] != 3:
                continue

            if only_complete:
                present = str(attrs.get("present_ENZ", "000"))
                if present != "111":
                    continue


            comp_order = str(attrs.get("component_order", "ENZ"))
            if comp_order != "ENZ":
                continue

            #'''
            if receiver_type is not None:
                rtype = str(attrs.get("receiver_type", ""))
                if rtype != receiver_type:
                    continue
            npts = data.shape[0]
            s0 = choose_window_start(attrs, npts, window_mode=window_mode, start_sample=start_sample)
            if s0 is None:
                s0=100
            s1 = s0 + 3000

            win = np.asarray(data[s0:s1, :], dtype=np.float32)
            if win.shape != (3000, 3):
                continue

            # ENZ -> ZNE -> (3, 3000)
            win = win[:, [2, 1, 0]].T

            x = np.zeros((3, 3000), dtype=np.float32)
            x[0] = zscore_safe(win[0])   # Z
            x[1] = zscore_safe(win[1])   # N
            x[2] = zscore_safe(win[2])   # E

            p_arrival_sample = attrs.get("p_arrival_sample", -1)
            s_arrival_sample = attrs.get("s_arrival_sample", -1)
            try:
                p_arrival_sample = int(p_arrival_sample)
            except Exception:
                p_arrival_sample = -1
            try:
                s_arrival_sample = int(s_arrival_sample)
            except Exception:
                s_arrival_sample = -1

            p_in_window = p_arrival_sample - s0 if p_arrival_sample >= 0 else -1
            s_in_window = s_arrival_sample - s0 if s_arrival_sample >= 0 else -1

            if not (0 <= p_in_window < 3000):
                p_in_window = -1
            if not (0 <= s_in_window < 3000):
                s_in_window = -1

            #if p_in_window < 0 or s_in_window < 0:
            #    continue

            y = build_seg_label(
                length=3000,
                p_idx=p_in_window,
                s_idx=s_in_window,
                half_width=label_half_width
            )

            X_list.append(x)
            Y_list.append(y)
            meta_list.append({
                "trace_name": trace_name,
                "split": trace_split,
                "station": str(attrs.get("receiver_code", "UNKNOWN")),
                "window_start_sample": int(s0),
                "window_end_sample": int(s1 - 1),
                "p_arrival_sample": int(p_arrival_sample),
                "s_arrival_sample": int(s_arrival_sample),
                "p_arrival_sample_in_window": int(p_in_window),
                "s_arrival_sample_in_window": int(s_in_window),
            })

    if len(X_list) == 0:
        X = np.empty((0, 3, 3000), dtype=np.float32)
        Y = np.empty((0, 3000), dtype=np.int64)
    else:
        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)

    return X, Y, meta_list


class H5SegDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def calc_sample_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)   # (B, L)
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        acc = calc_sample_accuracy(out.detach(), y)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}"
        })

    if n_batches == 0:
        return 0.0, 0.0

    return total_loss / n_batches, total_acc / n_batches


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)
            acc = calc_sample_accuracy(out, y)

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}"
            })

    if n_batches == 0:
        return 0.0, 0.0

    return total_loss / n_batches, total_acc / n_batches


def main():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    split_map = load_split_csv(SPLIT_CSV_PATH, trace_col=TRACE_COL, split_col=SPLIT_COL)

    print("Loading fine-tune train split from HDF5...")
    x_train, y_train, meta_train = load_h5_3c_100hz_training_data(
        H5_PATH,
        split_map=split_map,
        use_split="train",
        window_mode=WINDOW_MODE,
        start_sample=CUSTOM_START_SAMPLE,
        only_complete=ONLY_COMPLETE,
        receiver_type=RECEIVER_TYPE,
        label_half_width=LABEL_HALF_WIDTH,
    )

    print("Loading fine-tune val split from HDF5...")
    x_val, y_val, meta_val = load_h5_3c_100hz_training_data(
        H5_PATH,
        split_map=split_map,
        use_split="dev",
        window_mode=WINDOW_MODE,
        start_sample=CUSTOM_START_SAMPLE,
        only_complete=ONLY_COMPLETE,
        receiver_type=RECEIVER_TYPE,
        label_half_width=LABEL_HALF_WIDTH,
    )

    print("train x shape:", x_train.shape)
    print("train y shape:", y_train.shape)
    print("val x shape:", x_val.shape)
    print("val y shape:", y_val.shape)

    if x_train.shape[0] == 0:
        raise RuntimeError("No valid fine-tune training samples found in split=train.")
    if x_val.shape[0] == 0:
        raise RuntimeError("No valid fine-tune validation samples found in split=val.")

    train_set = H5SegDataset(x_train, y_train)
    val_set = H5SegDataset(x_val, y_val)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    sys.path.append(MODEL_DEF_PATH)

    device = torch.device(DEVICE)

    if method=='TF':
        from model_spec import ModelTF
        sys.path.pop()
    
    
        model = ModelTF(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    elif method=='LFB':
        from model_learnableFilter import ModelLFB
        sys.path.pop()
        model = ModelLFB(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    else:
        from model_str import Model
        sys.path.pop()
    
        model = Model(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    
    

    print(f"Loading pretrained weights from: {PRETRAINED_MODEL_PATH}")
    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    if USE_CLASS_WEIGHTS:
        weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)
    last_model_path = os.path.join(SAVE_DIR, LAST_MODEL_NAME)

    n_train = len(train_set)
    n_val = len(val_set)

    print(f"Start fine-tuning on {device} ...")
    print(f"train samples: {n_train}, val samples: {n_val}")

    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} | "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> saved best model: {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> no improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered: val_loss did not improve for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    torch.save(model.state_dict(), last_model_path)
    print(f"Saved last model: {last_model_path}")

    history_path = os.path.join(SAVE_DIR, HISTORY_JSON_NAME)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "settings": {
                "H5_PATH": H5_PATH,
                "SPLIT_CSV_PATH": SPLIT_CSV_PATH,
                "TRACE_COL": TRACE_COL,
                "SPLIT_COL": SPLIT_COL,
                "PRETRAINED_MODEL_PATH": PRETRAINED_MODEL_PATH,
                "WINDOW_MODE": WINDOW_MODE,
                "CUSTOM_START_SAMPLE": CUSTOM_START_SAMPLE,
                "ONLY_COMPLETE": ONLY_COMPLETE,
                "RECEIVER_TYPE": RECEIVER_TYPE,
                "LABEL_HALF_WIDTH": LABEL_HALF_WIDTH,
                "BATCH_SIZE": BATCH_SIZE,
                "NUM_EPOCHS": NUM_EPOCHS,
                "LEARNING_RATE": LEARNING_RATE,
                "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
                "CLASS_WEIGHTS": CLASS_WEIGHTS,
            },
            "num_train": n_train,
            "num_val": n_val,
            "history": history
        }, f, indent=2)

    print("Saved history:", history_path)
    print("Fine-tuning finished.")


if __name__ == "__main__":
    main()