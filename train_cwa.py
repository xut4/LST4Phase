import os
import sys
import json
import csv
import random
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# ========================= USER SETTINGS =========================

H5_PATH = "/home/YTX/STEAD/MERGE1n5_3000.hdf5"
META_CSV_PATH = "/home/YTX/STEAD/MERGE1n5_3000.csv"

H5_PATH = "/home/YTX/INSTANCE/MERGE.hdf5"
META_CSV_PATH = "/home/YTX/INSTANCE/MERGE.csv"

H5_PATH = "/home/YTX/TXED/TXED.hdf5"
META_CSV_PATH = "/home/YTX/TXED/TXED.csv"

H5_PATH = "/home/YTX/SeismoDual/MERGE_E45nN5_sb3000ZNE.hdf5"
META_CSV_PATH = "/home/YTX/SeismoDual/MERGE_E45nN5_sb3000ZNE.csv"

H5_PATH = "/home/YTX/CWA/MERGE_3000.hdf5"
META_CSV_PATH = "/home/YTX/CWA/MERGE_3000.csv"

SPLIT_CSV_PATH=META_CSV_PATH 
P_COL = "trace_p_arrival_sample"
S_COL = "trace_s_arrival_sample"
USE_SPLIT_CSV = True
PRETRAINED_MODEL_PATH = "./model/model_100Hz.pth"
PRETRAINED_MODEL_PATH = "./trainSTEAD_LSTCNN_PKWF_T0704/model_100hz_finetune_best.pth"
finetune=False

method='LSTCNN'#TF#WL#ST#LST#LSTCNN
use_bandpass=False
NOISE=True
LightLST=False
Ablation=False
 
TRACE_COL = "trace_name"
SPLIT_COL = "split"

CATEGORY_COL = "trace_category"
SNR_COL = "trace_snr_db"
P_STATUS_COL = "trace_p_status"
S_STATUS_COL = "trace_s_status"
STATION_COL = "station_code"  # optional

MODEL_DEF_PATH = "./model/"
SAVE_DIR = "./trainCWA_1to10in5_e3_T0810" 
test_csv_path=SAVE_DIR+"/test.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Window extraction from raw traces
# Options: "first30", "middle30", "custom", "p_center", "p_random"
WINDOW_MODE = "first30"
CUSTOM_START_SAMPLE = 0

# Data filtering
ONLY_VALID_PS = False           # True: require both P and S in metadata
MAX_SP = None                   # e.g. 5000, or None
REQUIRE_3C = True               # require 3 components after normalization

# Label generation
LABEL_HALF_WIDTH = 40
BACKGROUND_CLASS = 2            # P=0, S=1, BG=2


# SSTNet spectrum supervision
SST_LAMBDA = 0.5
SST_N_FREQ = 64
SST_KERNEL_SIZE = 129
SST_TOPK_RATIO = 0.15
SST_SAMPLE_RATE = 100

LST_GATE_LAMBDA = 1e-4
# Training
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
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

import subprocess
import threading
import csv

try:
    import psutil
except ImportError:
    psutil = None
class ResourceMonitor:
    def __init__(self, save_dir, gpuid=None, sample_interval=0.2):
        super().__init__()
        self.save_dir = save_dir
        self.gpuid = gpuid
        self.sample_interval = sample_interval

        self.csv_path = os.path.join(save_dir, "train_resource_log.csv")
        self._stop_event = None
        self._thread = None
        self._gpu_samples = []
        self._ram_samples = []
        self._epoch_start = None

        self.process = psutil.Process(os.getpid()) if psutil is not None else None

    def _query_gpu_mem_mb(self):
        try:
            if self.gpuid is not None:
                cmd = [
                    "nvidia-smi",
                    f"--id={self.gpuid}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
                return float(out.splitlines()[0])
            else:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
                return float(out.splitlines()[0])
        except Exception:
            return None

    def _poll_memory(self):
        while not self._stop_event.is_set():
            gpu_mem = self._query_gpu_mem_mb()
            if gpu_mem is not None:
                self._gpu_samples.append(gpu_mem)

            if self.process is not None:
                ram_mb = self.process.memory_info().rss / 1024**2
                self._ram_samples.append(ram_mb)

            time.sleep(self.sample_interval)
    def on_train_begin(self, model=None):
        if model is not None:
            total_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_count = total_count - trainable_count
    
            print("========== Model Size ==========")
            print(f"Total params: {total_count:,}")
            print(f"Trainable params: {trainable_count:,}")
            print(f"Non-trainable params: {non_trainable_count:,}")
    
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "epoch_time_sec",
                "peak_gpu_mem_mb",
                "peak_cpu_ram_mb"
            ])
            

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        self._gpu_samples = []
        self._ram_samples = []

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_memory)
        self._thread.daemon = True
        self._thread.start()

    def on_epoch_end(self, epoch, logs=None):
        self._stop_event.set()
        self._thread.join()

        epoch_time = time.time() - self._epoch_start
        peak_gpu = max(self._gpu_samples) if len(self._gpu_samples) > 0 else -1
        peak_ram = max(self._ram_samples) if len(self._ram_samples) > 0 else -1

        print(
            f"[Resource] epoch={epoch + 1} | "
            f"time={epoch_time:.2f}s | "
            f"peak_gpu_mem={peak_gpu:.2f} MB | "
            f"peak_cpu_ram={peak_ram:.2f} MB"
        )

        if logs is not None:
            logs["epoch_time_sec"] = epoch_time
            logs["peak_gpu_mem_mb"] = peak_gpu
            logs["peak_cpu_ram_mb"] = peak_ram

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(epoch_time, 4),
                round(peak_gpu, 4),
                round(peak_ram, 4)
            ])    
def count_parameters(model, name="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"\n=== Parameter Count: {name} ===")
    print(f"Total parameters:        {total:,}")
    print(f"Trainable parameters:    {trainable:,}")
    print(f"Non-trainable parameters:{non_trainable:,}")

    return total, trainable, non_trainable

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from scipy.signal import butter, filtfilt

def bandpass_filter_1d(x, fs=100.0, fmin=1.0, fmax=10.0, order=4):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x).astype(np.float32)
    
def zscore_safe(x):
    x = np.asarray(x, dtype=np.float32)
    std = np.std(x)
    if std == 0 or np.isnan(std):
        return x - np.mean(x)
    return (x - np.mean(x)) / std


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


def normalize_3c_shape(wf):
    wf = np.asarray(wf)
    if wf.ndim != 2:
        raise ValueError(f"Unexpected shape: {wf.shape}")
    if wf.shape[0] == 3 and wf.shape[1] != 3:
        wf = wf.T
    if wf.shape[1] != 3:
        raise ValueError(f"Need 3 channels, got {wf.shape}")
    return wf.astype(np.float32)


def build_seg_label(length, p_idx, s_idx, half_width=20):
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


def choose_window_start_random_ps(p, s, npts, min_p=300, max_p=900):
    if npts < 3000:
        return None
    if p is None or s is None or p < 0 or s < 0:
        return None

    try:
        p = int(p)
        s = int(s)
    except Exception:
        return None

    target_p = np.random.randint(min_p, max_p + 1)
    s0 = p - target_p
    s0 = max(0, min(s0, npts - 3000))

    p_in = p - s0
    s_in = s - s0
    if not (0 <= p_in < 3000 and 0 <= s_in < 3000):
        return None

    return int(s0)


def choose_window_start(p, s, npts, window_mode="first30", start_sample=0):
    if npts < 3000:
        return None

    if window_mode == "first30":
        #print(p)
        if p is None or p < 0:
            if NOISE:return 0
            return None###
        s0 = 0
    elif window_mode == "middle30":
        s0 = max(0, (npts - 3000) // 2)
    elif window_mode == "custom":
        s0 = int(start_sample)
    elif window_mode == "p_center":
        if p is None or p < 0:
            return None
        s0 = int(p) - 500
    elif window_mode == "p_random":
        if p is None or p < 0:
            #print(p)
            if NOISE:return 0
            return None###
        s0 = choose_window_start_random_ps(p, s, npts)
        if s0 is None:
            target_p = np.random.randint(300, 901)
            s0 = int(p) - target_p
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")

    s0 = max(0, min(int(s0), npts - 3000))
    return int(s0)


def build_split_lists_from_metadata(df, seed=42, train_ratio=0.8, dev_ratio=0.1):
    trace_names = df[TRACE_COL].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(trace_names)
    #    print(trace_names)
    n = len(trace_names)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_names = trace_names[:n_train]
    dev_names = trace_names[n_train:n_train + n_dev]
    test_names = trace_names[n_train + n_dev:]

    return {
        "train": set(train_names),
        "dev": set(dev_names),
        "test": set(test_names),
    }

def collect_valid_trace_names(
    h5_path,
    meta_df,
    use_split,
    split_map=None,
    use_split_csv=False,
    return_skip_stats=False,
):
    if use_split not in {"train", "dev", "test"}:
        raise ValueError(f"use_split must be train/dev/test, got: {use_split}")

    if use_split_csv:
        allowed = None
    else:
        split_sets = build_split_lists_from_metadata(meta_df, seed=SEED)
        allowed = split_sets[use_split]
        #print(allowed)
    valid_trace_names = []

    skip_stats = {
        "total_rows": 0,
        "wrong_split": 0,
        "trace_not_in_h5": 0,
        "missing_ps_when_required": 0,
        "bad_p_value": 0,
        "bad_s_value": 0,
        "sp_out_of_range": 0,
        "bad_dataset_ndim": 0,
        "not_3c": 0,
        "window_start_none": 0,
        "window_exceeds_length": 0,
        "accepted": 0,
    }

    with h5py.File(h5_path, "r") as f:
        g = f["data"] if "data" in f else f

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"Indexing {use_split}", leave=False):
            skip_stats["total_rows"] += 1
            trace_name = str(row[TRACE_COL])

            if use_split_csv:
                trace_split = split_map.get(trace_name, None)
                if trace_split != use_split:
                    skip_stats["wrong_split"] += 1
                    continue
            else:
                #print(trace_name)
                if trace_name not in allowed:
                    skip_stats["wrong_split"] += 1
                    continue

            if trace_name not in g:
                skip_stats["trace_not_in_h5"] += 1
                continue

            p_val = row.get(P_COL, np.nan)
            s_val = row.get(S_COL, np.nan)

            if ONLY_VALID_PS and (pd.isna(p_val) or pd.isna(s_val)):
                skip_stats["missing_ps_when_required"] += 1
                continue

            try:
                p_idx = int(p_val) if not pd.isna(p_val) else -1
            except Exception:
                p_idx = -1
                skip_stats["bad_p_value"] += 1

            try:
                s_idx = int(s_val) if not pd.isna(s_val) else -1
            except Exception:
                s_idx = -1
                skip_stats["bad_s_value"] += 1

            if MAX_SP is not None and p_idx >= 0 and s_idx >= 0:
                if s_idx - p_idx < 0 or s_idx - p_idx > MAX_SP:
                    skip_stats["sp_out_of_range"] += 1
                    continue

            ds = g[trace_name]
            shape = ds.shape
            if len(shape) != 2:
                skip_stats["bad_dataset_ndim"] += 1
                continue

            try:
                npts = max(shape[0], shape[1])
                #print(npts)
                if min(shape[0], shape[1]) != 3 and REQUIRE_3C:
                    skip_stats["not_3c"] += 1
                    continue
            except Exception:
                skip_stats["bad_dataset_ndim"] += 1
                continue

            s0 = choose_window_start(
                p_idx,
                s_idx,
                npts,
                window_mode=WINDOW_MODE,
                start_sample=CUSTOM_START_SAMPLE,
            )
            if s0 is None:
                skip_stats["window_start_none"] += 1
                continue

            s1 = s0 + 3000
            if s1 > npts:
                skip_stats["window_exceeds_length"] += 1
                continue

            valid_trace_names.append(trace_name)
            skip_stats["accepted"] += 1

    print(f"\n[Skip stats: {use_split}]")
    for k, v in skip_stats.items():
        print(f"{k}: {v}")

    if return_skip_stats:
        return valid_trace_names, skip_stats
    return valid_trace_names
    
def collect_valid_trace_names_old(
    h5_path,
    meta_df,
    use_split,
    split_map=None,
    use_split_csv=False,
):
    if use_split not in {"train", "dev", "test"}:
        raise ValueError(f"use_split must be train/dev/test, got: {use_split}")

    if use_split_csv:
        allowed = None
    else:
        split_sets = build_split_lists_from_metadata(meta_df, seed=SEED)
        allowed = split_sets[use_split]

    valid_trace_names = []

    with h5py.File(h5_path, "r") as f:
        g = f["data"] if "data" in f else f

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"Indexing {use_split}", leave=False):
            trace_name = str(row[TRACE_COL])

            if use_split_csv:
                trace_split = split_map.get(trace_name, None)
                if trace_split != use_split:
                    continue
            else:
                if trace_name not in allowed:
                    continue

            if trace_name not in g:
                continue

            p_val = row.get(P_COL, np.nan)
            s_val = row.get(S_COL, np.nan)

            if ONLY_VALID_PS and (pd.isna(p_val) or pd.isna(s_val)):
                continue

            try:
                p_idx = int(p_val) if not pd.isna(p_val) else -1
            except Exception:
                p_idx = -1
            try:
                s_idx = int(s_val) if not pd.isna(s_val) else -1
            except Exception:
                s_idx = -1

            if MAX_SP is not None and p_idx >= 0 and s_idx >= 0:
                if s_idx - p_idx < 0 or s_idx - p_idx > MAX_SP:
                    continue

            ds = g[trace_name]
            shape = ds.shape
            if len(shape) != 2:
                continue

            try:
                # only inspect small metadata-ish facts here
                npts = max(shape[0], shape[1])
                if min(shape[0], shape[1]) != 3 and REQUIRE_3C:
                    continue
            except Exception:
                continue

            s0 =choose_window_start(
                p_idx,
                s_idx,
                npts,
                window_mode=WINDOW_MODE,
                start_sample=CUSTOM_START_SAMPLE,
            )
            if s0 is None:
                continue

            s1 = s0 + 3000
            if s1 > npts:
                continue

            valid_trace_names.append(trace_name)

    return valid_trace_names


class RawCwaSegDataset(Dataset):
    def __init__(
        self,
        h5_path,
        meta_df,
        trace_names,
        window_mode="first30",
        start_sample=0,
        label_half_width=20,
    ):
        self.h5_path = h5_path
        self.meta_df = meta_df.copy()
        self.trace_names = trace_names
        self.window_mode = window_mode
        self.start_sample = start_sample
        self.label_half_width = label_half_width
        self._h5 = None

        self.meta_map = {}
        for _, row in self.meta_df.iterrows():
            self.meta_map[str(row[TRACE_COL])] = row.to_dict()

    def __len__(self):
        return len(self.trace_names)

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        row = self.meta_map[trace_name]

        f = self._get_h5()
        g = f["data"] if "data" in f else f
        ds = g[trace_name]

        wf = ds[()]
        wf = normalize_3c_shape(wf)
        npts = wf.shape[0]

        p_val = row.get(P_COL, np.nan)
        s_val = row.get(S_COL, np.nan)

        try:
            p_idx = int(p_val) if not pd.isna(p_val) else -1
        except Exception:
            p_idx = -1
        try:
            s_idx = int(s_val) if not pd.isna(s_val) else -1
        except Exception:
            s_idx = -1

        s0 = choose_window_start(
            p_idx,
            s_idx,
            npts,
            window_mode=self.window_mode,
            start_sample=self.start_sample,
        )
        if s0 is None:
            raise RuntimeError(f"Invalid window for trace {trace_name}")

        s1 = s0 + 3000
        win = np.asarray(wf[s0:s1, :], dtype=np.float32)
        if win.shape != (3000, 3):
            raise RuntimeError(f"Bad shape for {trace_name}: {win.shape}")

        # raw format assumed ENZ -> model input ZNE
        ##win = win[:, [2, 1, 0]].T
        win = win.T
        z = win[0]
        n = win[1]
        e = win[2]
        
        if use_bandpass:
            z = bandpass_filter_1d(z, fs=100.0, fmin=1.0, fmax=45.0, order=4)
            n = bandpass_filter_1d(n, fs=100.0, fmin=1.0, fmax=45.0, order=4)
            e = bandpass_filter_1d(e, fs=100.0, fmin=1.0, fmax=45.0, order=4)
        
        x = np.zeros((3, 3000), dtype=np.float32)
        x[0] = zscore_safe(z)####################normalize
        x[1] = zscore_safe(n)
        x[2] = zscore_safe(e)
        '''x[0] = z####################normalize
        x[1] = n
        x[2] = e'''
        p_in_window = p_idx - s0 if p_idx >= 0 else -1
        s_in_window = s_idx - s0 if s_idx >= 0 else -1

        if not (0 <= p_in_window < 3000):
            p_in_window = -1
        if not (0 <= s_in_window < 3000):
            s_in_window = -1

        y = build_seg_label(
            length=3000,
            p_idx=p_in_window,
            s_idx=s_in_window,
            half_width=self.label_half_width,
        )

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def calc_sample_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    correct = (pred == target).float().sum()
    total = target.numel()
    return (correct / total).item()

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    method=None,
    target_generator=None,
    lst_gate_lambda=0.0,
):
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_sst_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if method == "LST":
            out = model(x,ablation=Ablation)
        
            loss_task = criterion(out, y)
        
            # LST frequency-gate sparsity regularization
            if hasattr(model, "tf_branch") and hasattr(model.tf_branch, "freq_gate"):
                gate = torch.sigmoid(model.tf_branch.freq_gate)
                loss_sst = gate.mean()
                loss = loss_task + lst_gate_lambda * loss_sst
            else:
                loss_sst = torch.tensor(0.0, device=device)
                loss = loss_task
        
        else:
            out = model(x)
            loss_task = criterion(out, y)
            loss_sst = torch.tensor(0.0, device=device)
            loss = loss_task
        loss.backward()
        optimizer.step()

        acc = calc_sample_accuracy(out.detach(), y)

        total_loss += loss.item()
        total_task_loss += loss_task.item()
        total_sst_loss += loss_sst.item()
        total_acc += acc
        n_batches += 1

        if method == "LST":
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "task": f"{loss_task.item():.4f}",
                "gate": f"{loss_sst.item():.4f}",
                "acc": f"{acc:.4f}",
            })
        else:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
            })

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / n_batches,
        total_acc / n_batches,
        total_task_loss / n_batches
    )
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    method=None,
    target_generator=None,
    lst_gate_lambda=0.0
):
    model.eval()

    total_loss = 0.0
    total_task_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if  method == "LST":
                out = model(x,ablation=Ablation)            
                loss_task = criterion(out, y)
            
                # LST frequency-gate sparsity regularization
                if hasattr(model, "tf_branch") and hasattr(model.tf_branch, "freq_gate"):
                    gate = torch.sigmoid(model.tf_branch.freq_gate)
                    loss_sst = gate.mean()
                    loss = loss_task + lst_gate_lambda * loss_sst
                else:
                    loss_sst = torch.tensor(0.0, device=device)
                    loss = loss_task
            else:
                out = model(x)
                loss_task = criterion(out, y)
                loss_sst = torch.tensor(0.0, device=device)
                loss = loss_task

            acc = calc_sample_accuracy(out, y)

            total_loss += loss.item()
            total_task_loss += loss_task.item()
            total_sst_loss += loss_sst.item()
            total_acc += acc
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
            })

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / n_batches,
        total_acc / n_batches,
        total_task_loss / n_batches
    )
import time
def main():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    meta_df = pd.read_csv(META_CSV_PATH)
    if TRACE_COL not in meta_df.columns:
        raise KeyError(f"Metadata CSV missing column: {TRACE_COL}")

    split_map = None
    if USE_SPLIT_CSV:
        split_map = load_split_csv(SPLIT_CSV_PATH, trace_col=TRACE_COL, split_col=SPLIT_COL)
    train_trace_names, train_skip_stats = collect_valid_trace_names(
        H5_PATH,
        meta_df=meta_df,
        use_split="train",
        split_map=split_map,
        use_split_csv=USE_SPLIT_CSV,
        return_skip_stats=True,
    )
    
    val_trace_names, val_skip_stats = collect_valid_trace_names(
        H5_PATH,
        meta_df=meta_df,
        use_split="dev",
        split_map=split_map,
        use_split_csv=USE_SPLIT_CSV,
        return_skip_stats=True,
    )
    
    test_trace_names, test_skip_stats = collect_valid_trace_names(
        H5_PATH,
        meta_df=meta_df,
        use_split="test",
        split_map=split_map,
        use_split_csv=USE_SPLIT_CSV,
        return_skip_stats=True,
    )
    print("num train traces:", len(train_trace_names))
    print("num val traces:", len(val_trace_names))

    
    print("num test traces:", len(test_trace_names))
    
    test_df = meta_df[meta_df[TRACE_COL].astype(str).isin(test_trace_names)].copy()
    test_csv_path = os.path.join(SAVE_DIR, "test.csv")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved test csv: {test_csv_path}")
    
    
    if len(train_trace_names) == 0:
        raise RuntimeError("No valid training samples found in split=train.")
    if len(val_trace_names) == 0:
        raise RuntimeError("No valid validation samples found in split=dev.")

    train_set = RawCwaSegDataset(
        h5_path=H5_PATH,
        meta_df=meta_df,
        trace_names=train_trace_names,
        window_mode=WINDOW_MODE,
        start_sample=CUSTOM_START_SAMPLE,
        label_half_width=LABEL_HALF_WIDTH,
    )
    val_set = RawCwaSegDataset(
        h5_path=H5_PATH,
        meta_df=meta_df,
        trace_names=val_trace_names,
        window_mode=WINDOW_MODE,
        start_sample=CUSTOM_START_SAMPLE,
        label_half_width=LABEL_HALF_WIDTH,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    sys.path.append(MODEL_DEF_PATH)
    device = torch.device(DEVICE)  
    target_generator = None
    sst_loss_fn = None
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
    elif method=='WL':
        from model_spec import ModelWavelet
        sys.path.pop()
        model = ModelWavelet(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    elif method=='ST':
        from model_spec import ModelST
        sys.path.pop()
        model = ModelST(
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
    elif method=='BP':
        from model_BandPass import ModelBP
        sys.path.pop()
        model = ModelBP(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    elif method=='LST':
        from model_sstnet import ModelLST
        sys.path.pop()
        model = ModelLST(
            in_length=100 * 30,
            in_channels=3,
            Lightmode=LightLST,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    elif method=='LSTCNN':
        from model_sstnet import ModelLSTCNNViT
        sys.path.pop()
        model = ModelLSTCNNViT(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    elif method=='LST2D':
        from model_sstnet import ModelLST2D
        sys.path.pop()
        model = ModelLST2D(
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
    
    count_parameters(model, method)
    monitor = ResourceMonitor(
        save_dir=SAVE_DIR,
        gpuid=None,         
        sample_interval=0.05
    )

    monitor.on_train_begin(model)
    
    if finetune:
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

    print(f"Start training on {device} ...")
    print(f"train samples: {n_train}, val samples: {n_val}")

    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
    
        monitor.on_epoch_begin(epoch)
    
        train_loss, train_acc, train_task_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            method=method,
            target_generator=target_generator,
            lst_gate_lambda=LST_GATE_LAMBDA,
        )
    
        monitor.on_epoch_end(epoch)
        if DEVICE == "cuda":
            peak_train_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_train_mem_mb = 0.0
        
        print(
            f"[Epoch {epoch+1}] "
            #f"time={epoch_time_sec:.2f}s, "
            f"peak_mem={peak_train_mem_mb:.1f}MB"
        )
        val_loss, val_acc, val_task_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            method=method,
            target_generator=target_generator,
            lst_gate_lambda=LST_GATE_LAMBDA,
        )
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
                "META_CSV_PATH": META_CSV_PATH,
                "USE_SPLIT_CSV": USE_SPLIT_CSV,
                "SPLIT_CSV_PATH": SPLIT_CSV_PATH,
                "TRACE_COL": TRACE_COL,
                "SPLIT_COL": SPLIT_COL,
                "P_COL": P_COL,
                "S_COL": S_COL,
                'PRETRAINED_MODEL_PATH' :PRETRAINED_MODEL_PATH,
                'finetune':finetune,
                'method':method,
                'use_bandpass':use_bandpass,
                'NOISE':NOISE,
                "WINDOW_MODE": WINDOW_MODE,
                "CUSTOM_START_SAMPLE": CUSTOM_START_SAMPLE,
                "ONLY_VALID_PS": ONLY_VALID_PS,
                "MAX_SP": MAX_SP,
                "LABEL_HALF_WIDTH": LABEL_HALF_WIDTH,
                "BATCH_SIZE": BATCH_SIZE,
                "NUM_EPOCHS": NUM_EPOCHS,
                "LEARNING_RATE": LEARNING_RATE,
                "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
                "CLASS_WEIGHTS": CLASS_WEIGHTS,
            },
            "num_train": n_train,
            "num_val": n_val,
            "history": history,
        }, f, indent=2)

    print("Saved history:", history_path)
    print("Training finished.")


if __name__ == "__main__":
    device = torch.device(DEVICE)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("DEVICE =", DEVICE)
    print("device =", device)
    main()