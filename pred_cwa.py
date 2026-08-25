import sys
import os
import csv
import json
import numpy as np
import h5py
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import threading
import time
try:
    from thop import profile, clever_format
    HAS_THOP = True
except Exception:
    HAS_THOP = False

# ========================= USER SETTINGS =========================

P_COL = "trace_p_arrival_sample"
S_COL = "trace_s_arrival_sample"

H5_PATH = "/home/YTX/STEAD/MERGE1n5_3000.hdf5"
META_CSV_PATH = "/home/YTX/STEAD/MERGE1n5_3000.csv"
TARGET_CSV_PATH = "./trainSTEAD1n5_LST_T06181413/test.csv"

H5_PATH = "/home/YTX/INSTANCE/MERGE.hdf5"
META_CSV_PATH = "/home/YTX/INSTANCE/MERGE.csv"
TARGET_CSV_PATH = "./trainINS_LSTCNN_e3_T0717/test.csv"

H5_PATH = "/home/YTX/TXED/TXED.hdf5"
META_CSV_PATH = "/home/YTX/TXED/TXED.csv"
TARGET_CSV_PATH = "./trainTXED_LST_T0719/test.csv"

H5_PATH = "/home/YTX/SeismoDual/MERGE_E45nN5_sb3000ZNE.hdf5"
META_CSV_PATH = "/home/YTX/SeismoDual/MERGE_E45nN5_sb3000ZNE.csv"
TARGET_CSV_PATH = "./train45/test.csv"

H5_PATH = "/home/YTX/CWA/MERGE_3000.hdf5"
META_CSV_PATH = "/home/YTX/CWA/MERGE_3000.csv"
TARGET_CSV_PATH = "./trainCWAmerge_T/test.csv"

TRACE_COL = "trace_name"
CATEGORY_COL = "trace_category"
SNR_COL = "trace_snr_db"
P_STATUS_COL = "trace_p_status"
S_STATUS_COL = "trace_s_status"
STATION_COL = "station_code"

MODEL_PATH = "./trainCWA_1to10in5_e3_T0810/model_100hz_finetune_best.pth"
#MODEL_PATH = "./model/model_100Hz.pth"
MODEL_DEF_PATH = "./model/"
method='LSTCNN'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_bandpass=False
NOISE=True 
LightLST=False
ablation=False#lst
 
OUTPUT_DIR = "./testCWA_1to10in5_e3_T0810"
# Options: "first30", "middle30", "custom", "p_center", "p_random"
WINDOW_MODE = "first30"
CUSTOM_START_SAMPLE = 0
# FixedSSTNet spectrum supervision
SST_N_FREQ = 64
SST_KERNEL_SIZE = 129
SST_TOPK_RATIO = 0.15
SST_SAMPLE_RATE = 100
# Prediction source mode:
# "full"     -> use full META_CSV_PATH
# "csv_list" -> only use traces listed in TARGET_CSV_PATH
PRED_SOURCE_MODE = "csv_list"

SAVE_FIG = False
SHOW_FIG = False
SAVE_CSV = True

MAX_PLOTS = None
TOLERANCE_SAMPLES = 50   # 100 Hz => 0.50 s
# ================================================================
import matplotlib.pyplot as plt
def evaluate_event_detection(pred_event, gt_event):
    if gt_event and pred_event:
        return {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    elif gt_event and not pred_event:
        return {"tp": 0, "fp": 0, "fn": 1, "tn": 0}
    elif (not gt_event) and pred_event:
        return {"tp": 0, "fp": 1, "fn": 0, "tn": 0}
    else:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 1}


def aggregate_detection_metrics(eval_rows):
    tp = sum(r["tp"] for r in eval_rows)
    fp = sum(r["fp"] for r in eval_rows)
    fn = sum(r["fn"] for r in eval_rows)
    tn = sum(r["tn"] for r in eval_rows)

    precision, recall, acc, fscore = calc_metrics(tp, fp, fn, tn)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "f1": float(fscore),
    }
def count_parameters(model, name="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== Parameter Count: {name} ===")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    return total, trainable

def query_gpu_mem_mb(gpuid=None):
    try:
        if gpuid is not None:
            cmd = [
                "nvidia-smi",
                f"--id={gpuid}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ]
        else:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"
            ]

        out = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            timeout=2
        ).decode().strip()

        vals = []
        for line in out.splitlines():
            line = line.strip()
            if line:
                vals.append(float(line))

        if len(vals) == 0:
            return None

        return max(vals)

    except Exception:
        return None

def benchmark_inference_memory_nvidia_smi_delta(
    model,
    device,
    input_shape=(1, 3, 3000),
    gpuid=None,
    runs=30,
    name="model"
):
    """
    Nvidia-smi based inference memory benchmark.

    This is designed to compare with Keras/TensorFlow nvidia-smi delta:
      baseline_gpu_mem_mb = memory before dummy inference
      peak_gpu_mem_mb     = peak memory during dummy inference
      delta_gpu_mem_mb    = peak - baseline

    Use delta_gpu_mem_mb for cross-framework comparison.
    """
    if device.type != "cuda":
        print(f"[WARN] CUDA not available. Skip nvidia-smi memory for {name}.")
        return 0.0, 0.0, 0.0

    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    dummy = torch.randn(*input_shape).to(device)

    torch.cuda.synchronize()

    baseline_gpu_mem_mb = query_gpu_mem_mb(gpuid)
    if baseline_gpu_mem_mb is None:
        baseline_gpu_mem_mb = -1.0

    monitor = NvidiaSmiMonitor(
        gpuid=gpuid,
        sample_interval=0.02
    )

    monitor.start()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)

    torch.cuda.synchronize()

    peak_gpu_mem_mb = monitor.stop()

    if baseline_gpu_mem_mb >= 0 and peak_gpu_mem_mb >= 0:
        delta_gpu_mem_mb = max(0.0, peak_gpu_mem_mb - baseline_gpu_mem_mb)
    else:
        delta_gpu_mem_mb = -1.0

    print(f"\n=== Inference Memory by nvidia-smi: {name} ===")
    print(f"Baseline GPU memory: {baseline_gpu_mem_mb:.2f} MB")
    print(f"Peak GPU memory:     {peak_gpu_mem_mb:.2f} MB")
    print(f"Delta GPU memory:    {delta_gpu_mem_mb:.2f} MB")

    return delta_gpu_mem_mb, peak_gpu_mem_mb, baseline_gpu_mem_mb
class NvidiaSmiMonitor:
    def __init__(self, gpuid=None, sample_interval=0.02):
        self.gpuid = gpuid
        self.sample_interval = sample_interval
        self.samples = []
        self._stop_event = None
        self._thread = None

    def _poll(self):
        while not self._stop_event.is_set():
            mem = query_gpu_mem_mb(self.gpuid)
            if mem is not None:
                self.samples.append(mem)
            time.sleep(self.sample_interval)

    def start(self):
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

        if len(self.samples) == 0:
            return -1.0

        return max(self.samples)
def profile_macs(model, device, input_shape=(1, 3, 3000), name="model"):
    if not HAS_THOP:
        print(f"[WARN] thop not installed. Skip MACs profiling for {name}.")
        return None, None

    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    try:
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        macs_fmt, params_fmt = clever_format([macs, params], "%.3f")

        print(f"\n=== MACs / Params: {name} ===")
        print(f"MACs:   {macs_fmt}")
        print(f"Params: {params_fmt}")

        return macs, params

    except Exception as e:
        print(f"[WARN] Failed to profile {name}: {e}")
        return None, None


def benchmark_inference_time(model, device, input_shape=(1, 3, 3000),
                             warmup=30, runs=100, name="model"):
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        for _ in range(runs):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    avg_ms = (end - start) / runs * 1000.0
    traces_per_sec = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"\n=== Inference Time: {name} ===")
    print(f"Average time: {avg_ms:.4f} ms / trace")
    print(f"Throughput:   {traces_per_sec:.2f} traces / sec")

    return avg_ms, traces_per_sec


def benchmark_inference_memory(model, device, input_shape=(1, 3, 3000), name="model"):
    if device.type != "cuda":
        print(f"[WARN] CUDA not available. Skip inference memory for {name}.")
        return 0.0, 0.0

    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(dummy)

    torch.cuda.synchronize()

    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    print(f"\n=== Inference Memory: {name} ===")
    print(f"Peak allocated: {peak_allocated_mb:.2f} MB")
    print(f"Peak reserved:  {peak_reserved_mb:.2f} MB")

    return peak_allocated_mb, peak_reserved_mb
    
def make_tf_specs_for_plot(x, pred_spec=None, target_generator=None, lst_spec=None):
    """
    x:        (1, 3, 3000)
    pred_spec: SSTNet spectrum, (1, 1, F, T)
    lst_spec:  LST spectrum,    (1, 1, F, T)
    """
    wave_np = x[0].detach().cpu().numpy()  # (3, 3000)

    wave_norm01 = minmax_01_per_channel(wave_np)
    wave_bp145 = bandpass_wave_1_45(wave_np, fs=100.0)

    # STFT
    z = x[0, 0]
    stft_complex = torch.stft(
        z,
        n_fft=128,
        hop_length=16,
        win_length=128,
        window=torch.hann_window(128).to(z.device),
        return_complex=True
    )
    stft_spec = torch.log1p(torch.abs(stft_complex)).detach().cpu().numpy()

    # Sparse S-transform target
    if target_generator is not None:
        with torch.no_grad():
            target_spec = target_generator(x)
        sparse_spec = target_spec[0, 0].detach().cpu().numpy()
    else:
        sparse_spec = None

    # SSTNet predicted spectrum
    if pred_spec is not None:
        sstnet_spec = pred_spec[0, 0].detach().cpu().numpy()
    else:
        sstnet_spec = None

    # LST spectrum
    if lst_spec is not None:
        lst_spec_np = lst_spec[0, 0].detach().cpu().numpy()
    else:
        lst_spec_np = None

    return wave_np, wave_norm01, wave_bp145, stft_spec, sparse_spec, lst_spec_np, sstnet_spec
'''def plot_tf(spec, title, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency bin")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.close()'''
import numpy as np
import matplotlib.pyplot as plt
def draw_pick_lines(ax, p_gt_idx, s_gt_idx, p_pick_idx, s_pick_idx):
    if p_gt_idx is not None and p_gt_idx >= 0:
        ax.axvline(p_gt_idx, color="green", linestyle="--", linewidth=1.1, label="P GT")
    if s_gt_idx is not None and s_gt_idx >= 0:
        ax.axvline(s_gt_idx, color="orange", linestyle="--", linewidth=1.1, label="S GT")
    if p_pick_idx is not None and p_pick_idx >= 0:
        ax.axvline(p_pick_idx, color="blue", linestyle="-", linewidth=1.0, label="P Pick")
    if s_pick_idx is not None and s_pick_idx >= 0:
        ax.axvline(s_pick_idx, color="red", linestyle="-", linewidth=1.0, label="S Pick")


def plot_wave_channels(ax, wave, title, p_gt_idx, s_gt_idx, p_pick_idx, s_pick_idx):
    """
    wave: (C, T)
    """
    n_samples = wave.shape[1]
    sample_axis = np.arange(n_samples)
    labels = ["Z", "N", "E"]

    for c in range(wave.shape[0]):
        label = labels[c] if c < len(labels) else f"Ch{c}"
        ax.plot(sample_axis, wave[c], linewidth=0.7, label=label)

    draw_pick_lines(ax, p_gt_idx, s_gt_idx, p_pick_idx, s_pick_idx)

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, n_samples - 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=4)

def plot_tf_map(ax, fig, spec, title, n_samples, p_gt_idx, s_gt_idx, p_pick_idx, s_pick_idx):
    if spec is None:
        ax.set_title(title + " (not available)")
        ax.axis("off")
        return

    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, n_samples - 1, 0, spec.shape[0]]
    )

    draw_pick_lines(ax, p_gt_idx, s_gt_idx, p_pick_idx, s_pick_idx)

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency Bin")
    ax.set_xlim(0, n_samples - 1)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
def plot_wave_tf_stack(
    wave,
    wave_norm01,
    wave_bp145,
    stft_spec,
    sparse_spec,
    lst_spec,
    sst_spec,
    title,
    save_path,
    p_gt_idx=-1,
    s_gt_idx=-1,
    p_pick_idx=-1,
    s_pick_idx=-1,
):
    """
    wave:        (C, T), z-score waveform used by model
    wave_norm01: (C, T), min-max normalized waveform
    wave_bp145:  (C, T), 1-45 Hz bandpass waveform
    stft_spec:   (F1, T1)
    sparse_spec: (F2, T2)
    sst_spec:    (F3, T3)

    X-axis unit: sample
    """
    n_samples = wave.shape[1]

    fig, axes = plt.subplots(7, 1, figsize=(15, 20), constrained_layout=True)
    plot_wave_channels(
        axes[0],
        wave,
        "Waveform Used by Model (Z-score)",
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )

    plot_wave_channels(
        axes[1],
        wave_norm01,
        "Waveform Normalized to 0-1",
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )

    plot_wave_channels(
        axes[2],
        wave_bp145,
        "Waveform Bandpass 1-45 Hz",
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )
    plot_tf_map(
        axes[3],
        fig,
        stft_spec,
        "STFT Spectrogram",
        n_samples,
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )
    
    plot_tf_map(
        axes[4],
        fig,
        sparse_spec,
        "Sparse S-transform Target",
        n_samples,
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )
    
    plot_tf_map(
        axes[5],
        fig,
        lst_spec,
        "LST Learnable S-transform-like Map",
        n_samples,
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )
    
    plot_tf_map(
        axes[6],
        fig,
        sst_spec,
        "SSTNet Predicted Spectrum",
        n_samples,
        p_gt_idx,
        s_gt_idx,
        p_pick_idx,
        s_pick_idx
    )

    fig.suptitle(title, fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
'''from scipy.signal import butter, filtfilt
def fuse_lst_sst(pred_lst, pred_sst, noise_threshold=0.8):
    """
    pred_lst: (3, T)
    pred_sst: (3, T)

    class 0: P
    class 1: S
    class 2: background/noise
    """
    pred_fused = np.zeros_like(pred_lst)

    # LST is stronger for P/S picking
    pred_fused[0] = 0.8 * pred_lst[0] + 0.2 * pred_sst[0]
    pred_fused[1] = 0.8 * pred_lst[1] + 0.2 * pred_sst[1]

    # SST is stronger for noise/background
    pred_fused[2] = 0.15 * pred_lst[2] + 0.85 * pred_sst[2]

    # Noise-aware suppression
    noise_gate = pred_sst[2]
    suppress_mask = noise_gate > noise_threshold

    pred_fused[0, suppress_mask] *= 0.7
    pred_fused[1, suppress_mask] *= 0.7
    pred_fused[2, suppress_mask] = np.maximum(
        pred_fused[2, suppress_mask],
        noise_gate[suppress_mask]
    )

    # Normalize probability
    pred_fused = pred_fused / (np.sum(pred_fused, axis=0, keepdims=True) + 1e-8)

    return pred_fused'''
    

def minmax_01_per_channel(wave):
    """
    wave: (C, T)
    return: (C, T), each channel normalized to 0~1
    """
    wave = np.asarray(wave, dtype=np.float32)
    out = np.zeros_like(wave, dtype=np.float32)

    for c in range(wave.shape[0]):
        x = wave[c]
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax - xmin < 1e-8:
            out[c] = 0.0
        else:
            out[c] = (x - xmin) / (xmax - xmin)

    return out


def bandpass_wave_1_45(wave, fs=100.0):
    """
    wave: (C, T)
    return: bandpass filtered wave, 1~45 Hz
    """
    wave = np.asarray(wave, dtype=np.float32)
    out = np.zeros_like(wave, dtype=np.float32)

    for c in range(wave.shape[0]):
        out[c] = bandpass_filter(
            wave[c],
            fs=fs,
            fmin=1.0,
            fmax=45.0,
            order=4
        )

    return out
def zscore_safe(x):
    x = np.asarray(x, dtype=np.float32)
    std = np.std(x)
    if std == 0 or np.isnan(std):
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def safe_name(s):
    s = str(s)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        s = s.replace(ch, '_')
    return s


def calc_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, acc, fscore


def normalize_3c_shape(wf):
    wf = np.asarray(wf)
    if wf.ndim != 2:
        raise ValueError(f"Unexpected shape: {wf.shape}")
    if wf.shape[0] == 3 and wf.shape[1] != 3:
        wf = wf.T
    if wf.shape[1] != 3:
        raise ValueError(f"Need 3 channels, got {wf.shape}")
    return wf.astype(np.float32)


def load_metadata_csv(meta_csv_path):
    df = pd.read_csv(meta_csv_path)
    if TRACE_COL not in df.columns:
        raise KeyError(f"CSV missing trace column: {TRACE_COL}")
    return df


def load_target_csv(target_csv_path):
    df = pd.read_csv(target_csv_path)
    if TRACE_COL not in df.columns:
        raise KeyError(f"Target CSV missing column: {TRACE_COL}")
    return df


def ensure_required_columns(df):
    """
    Make sure downstream code always finds these columns.
    If they don't exist, create them with safe default values.
    """
    required_defaults = {
        TRACE_COL: "",
        P_COL: -1,
        S_COL: -1,
        CATEGORY_COL: "",
        SNR_COL: "",
        P_STATUS_COL: "",
        S_STATUS_COL: "",
        STATION_COL: "UNKNOWN",
    }

    out = df.copy()
    for col, default_val in required_defaults.items():
        if col not in out.columns:
            out[col] = default_val
    return out


def build_prediction_meta_df(meta_df, pred_source_mode="full", target_csv_path=None):
    """
    Two modes:
    1) full:
       use full metadata csv
    2) csv_list:
       use TARGET_CSV_PATH as the testing list
       - ignore split column entirely
       - if target csv already contains metadata cols, use them directly
       - if target csv only contains trace_name, merge from meta_df
    """
    meta_df = ensure_required_columns(meta_df)

    if pred_source_mode == "full":
        return meta_df.copy()

    if pred_source_mode == "csv_list":
        if target_csv_path is None:
            raise ValueError("target_csv_path is required when pred_source_mode='csv_list'")

        target_df = load_target_csv(target_csv_path)
        target_df = ensure_required_columns(target_df)

        # Ignore split column entirely, even if it exists and is wrong.
        # Whole target CSV is treated as the test list.

        target_trace_names = (
            target_df[TRACE_COL]
            .astype(str)
            .fillna("")
            .tolist()
        )

        if len(target_trace_names) == 0:
            raise RuntimeError("TARGET_CSV_PATH is empty.")

        target_has_full_metadata = all(
            col in target_df.columns
            for col in [P_COL, S_COL, CATEGORY_COL, SNR_COL, P_STATUS_COL, S_STATUS_COL, STATION_COL]
        )

        if target_has_full_metadata:
            # Directly use target csv as prediction metadata
            pred_meta_df = target_df.copy()
        else:
            # Only use trace_name list, merge needed metadata from meta_df
            meta_df_keyed = meta_df.copy()
            meta_df_keyed[TRACE_COL] = meta_df_keyed[TRACE_COL].astype(str)

            pred_meta_df = meta_df_keyed[
                meta_df_keyed[TRACE_COL].isin(set(map(str, target_trace_names)))
            ].copy()

            if len(pred_meta_df) == 0:
                raise RuntimeError(
                    "No matched trace_name found between TARGET_CSV_PATH and META_CSV_PATH."
                )
        ordered_trace_names = pd.Index(target_trace_names).drop_duplicates()

        # Keep target csv order
        pred_meta_df[TRACE_COL] = pred_meta_df[TRACE_COL].astype(str)
        pred_meta_df["_trace_order"] = pd.Categorical(
            pred_meta_df[TRACE_COL],
            categories=list(map(str, ordered_trace_names)),
            ordered=True
        )
        pred_meta_df = pred_meta_df.sort_values("_trace_order").drop(columns=["_trace_order"])

        # Drop duplicate trace names, keep first occurrence in target order
        pred_meta_df = pred_meta_df.drop_duplicates(subset=[TRACE_COL], keep="first").reset_index(drop=True)

        return pred_meta_df

    raise ValueError(f"Unknown pred_source_mode: {pred_source_mode}")


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
    #prin#t(s0)
    p_in = p - s0
    s_in = s - s0
    if not (0 <= p_in < 3000 and 0 <= s_in < 3000):
        return None

    return int(s0)


def choose_window_start(p, s, npts, window_mode="first30", start_sample=0):
    if npts < 3000:
        return None

    if window_mode == "first30":
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
        s0 = int(p) - 700
    elif window_mode == "p_random":
        if p is None or p < 0:
            return None
        s0 = choose_window_start_random_ps(p, s, npts)
        if s0 is None:
            return None
         #   target_p = np.random.randint(300, 901)
         # s0 = int(p) - target_p
            
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")

    s0 = max(0, min(int(s0), npts - 3000))
    return int(s0)

from collections import Counter

def iter_raw_h5_csv_for_pred(
    h5_path,
    meta_df,
    window_mode="first30",
    start_sample=0,
    return_skip_counter=False,
):
    meta_df = ensure_required_columns(meta_df)
    skip = Counter()

    def gen():
        nonlocal skip

        with h5py.File(h5_path, "r") as f:
            g = f["data"] if "data" in f else f

            for _, row in meta_df.iterrows():
                trace_name = str(row[TRACE_COL])

                if trace_name not in g:
                    skip["missing_in_h5"] += 1
                    continue

                try:
                    wf = g[trace_name][()]
                    raw_shape = wf.shape
                    wf = normalize_3c_shape(wf)
                except Exception as e:
                    skip[f"bad_waveform_shape_or_read:{type(e).__name__}"] += 1
                    continue

                npts = wf.shape[0]
                if npts < 3000:
                    skip["npts_lt_3000"] += 1
                    continue

                p_arrival_sample = row.get(P_COL, -1)
                s_arrival_sample = row.get(S_COL, -1)

                try:
                    p_arrival_sample = int(p_arrival_sample) if not pd.isna(p_arrival_sample) else -1
                except Exception:
                    p_arrival_sample = -1

                try:
                    s_arrival_sample = int(s_arrival_sample) if not pd.isna(s_arrival_sample) else -1
                except Exception:
                    s_arrival_sample = -1
                #if p_arrival_sample==-1:print('-1')
                s0 = choose_window_start(
                    p_arrival_sample,
                    s_arrival_sample,
                    npts,
                    window_mode=window_mode,
                    start_sample=start_sample,
                )
                if s0 is None:
                    skip["bad_window_start"] += 1
                    continue

                s1 = s0 + 3000
                if s1 > npts:
                    skip["window_exceeds_npts"] += 1
                    continue

                win = np.asarray(wf[s0:s1, :], dtype=np.float32)
                if win.shape != (3000, 3):
                    skip[f"bad_window_shape:{win.shape}"] += 1
                    continue

                win = win.T

                z = win[0]
                n = win[1]
                e = win[2]
                
                if use_bandpass:
                    z = bandpass_filter(z, fs=100.0, fmin=1.0, fmax=45.0, order=4)
                    n = bandpass_filter(n, fs=100.0, fmin=1.0, fmax=45.0, order=4)
                    e = bandpass_filter(e, fs=100.0, fmin=1.0, fmax=45.0, order=4)
                
                out = np.zeros((3, 3000), dtype=np.float32)
                out[0] = zscore_safe(z)
                out[1] = zscore_safe(n)###normalize
                out[2] = zscore_safe(e)
                '''out[0] = z
                out[1] = n
                out[2] = e'''
                
                '''out = np.zeros((3, 3000), dtype=np.float32)
                out[0] = zscore_safe(win[0])
                out[1] = zscore_safe(win[1])
                out[2] = zscore_safe(win[2])'''
                #print(p_arrival_sample , s0)
                p_in_window = p_arrival_sample - s0 if p_arrival_sample >= 0 else -1
                s_in_window = s_arrival_sample - s0 if s_arrival_sample >= 0 else -1

                if not (0 <= p_in_window < 3000):
                    p_in_window = -1
                if not (0 <= s_in_window < 3000):
                    s_in_window = -1

                station = str(row.get(STATION_COL, "UNKNOWN"))

                meta = {
                    "window_start_sample": int(s0),
                    "window_end_sample": int(s1 - 1),
                    "p_arrival_sample": int(p_arrival_sample),
                    "s_arrival_sample": int(s_arrival_sample),
                    "p_arrival_sample_in_window": int(p_in_window),
                    "s_arrival_sample_in_window": int(s_in_window),
                    "trace_category": row.get(CATEGORY_COL, ""),
                    "trace_snr_db": row.get(SNR_COL, ""),
                    "trace_p_status": row.get(P_STATUS_COL, ""),
                    "trace_s_status": row.get(S_STATUS_COL, ""),
                }

                skip["processed"] += 1
                yield station, trace_name, out, meta

    if return_skip_counter:
        return gen(), skip
    return gen()
'''def iter_raw_h5_csv_for_pred_oldversion(
    h5_path,
    meta_df,
    window_mode="first30",
    start_sample=0,
):
    meta_df = ensure_required_columns(meta_df)

    with h5py.File(h5_path, "r") as f:
        g = f["data"] if "data" in f else f

        for _, row in meta_df.iterrows():
            trace_name = str(row[TRACE_COL])

            if trace_name not in g:
                continue

            try:
                wf = g[trace_name][()]
                wf = normalize_3c_shape(wf)
            except Exception:
                continue

            npts = wf.shape[0]
            if npts < 3000:
                continue

            p_arrival_sample = row.get(P_COL, -1)
            s_arrival_sample = row.get(S_COL, -1)

            try:
                p_arrival_sample = int(p_arrival_sample) if not pd.isna(p_arrival_sample) else -1
            except Exception:
                p_arrival_sample = -1

            try:
                s_arrival_sample = int(s_arrival_sample) if not pd.isna(s_arrival_sample) else -1
            except Exception:
                s_arrival_sample = -1

            s0 = choose_window_start(
                p_arrival_sample,
                s_arrival_sample,
                npts,
                window_mode=window_mode,
                start_sample=start_sample,
            )
            if s0 is None:
                continue

            s1 = s0 + 3000
            if s1 > npts:
                continue

            win = np.asarray(wf[s0:s1, :], dtype=np.float32)
            if win.shape != (3000, 3):
                continue

            # raw format assumed ENZ -> model input ZNE
            # win = win[:, [2, 1, 0]].T
            win = win.T

            out = np.zeros((3, 3000), dtype=np.float32)
            out[0] = zscore_safe(win[0])   # Z
            out[1] = zscore_safe(win[1])   # N
            out[2] = zscore_safe(win[2])   # E

            p_in_window = p_arrival_sample - s0 if p_arrival_sample >= 0 else -1
            s_in_window = s_arrival_sample - s0 if s_arrival_sample >= 0 else -1

            if not (0 <= p_in_window < 3000):
                p_in_window = -1
            if not (0 <= s_in_window < 3000):
                s_in_window = -1

            station = str(row.get(STATION_COL, "UNKNOWN"))

            meta = {
                "window_start_sample": int(s0),
                "window_end_sample": int(s1 - 1),
                "p_arrival_sample": int(p_arrival_sample),
                "s_arrival_sample": int(s_arrival_sample),
                "p_arrival_sample_in_window": int(p_in_window),
                "s_arrival_sample_in_window": int(s_in_window),
                "trace_category": row.get(CATEGORY_COL, ""),
                "trace_snr_db": row.get(SNR_COL, ""),
                "trace_p_status": row.get(P_STATUS_COL, ""),
                "trace_s_status": row.get(S_STATUS_COL, ""),
            }

            yield station, trace_name, out, meta
'''

def bandpass_filter(x, fs=100.0, fmin=1.0, fmax=30.0, order=4):
    x = np.asarray(x, dtype=np.float32)

    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq

    if low <= 0:
        low = 1e-6
    if high >= 1:
        high = 0.999

    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, x)
    return y


def plot_wave(ax, lapse_time, wave, best_p_idx, best_s_idx, sf, ylabel, p_gt_idx=None, s_gt_idx=None):
    ax.plot(lapse_time, wave, c="black", lw=0.6)

    if best_p_idx is not None and best_p_idx >= 0:
        ax.axvline(x=best_p_idx * sf, c="blue", lw=0.8)

    if best_s_idx is not None and best_s_idx >= 0:
        ax.axvline(x=best_s_idx * sf, c="red", lw=0.8)

    if p_gt_idx is not None and p_gt_idx >= 0:
        ax.axvline(x=p_gt_idx * sf, c="green", lw=1.2, linestyle="--")

    if s_gt_idx is not None and s_gt_idx >= 0:
        ax.axvline(x=s_gt_idx * sf, c="orange", lw=1.2, linestyle="--")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, 30)


def plot_pred(ax, lapse_time, pred, best_idx, phase, sf):
    if phase == "P":
        ylabel = "P-wave Pred."
        c = "blue"
    elif phase == "S":
        ylabel = "S-wave Pred."
        c = "red"
    else:
        ylabel = "Pred."
        c = "black"

    ax.plot(lapse_time, pred, c=c, lw=0.8, label=ylabel)

    if best_idx is not None and best_idx >= 0:
        ax.axvline(x=best_idx * sf, c=c, lw=0.8)

    ax.set_xlim(0, 30)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)


def summarize_prediction(pred, sf=0.01):
    pidxs, pinfo = find_peaks(pred[0, :], distance=int(1 / sf), height=0.5)
    sidxs, sinfo = find_peaks(pred[1, :], distance=int(1 / sf), height=0.5)

    if len(pidxs) > 0:
        best_p_i = int(np.argmax(pinfo["peak_heights"]))
        best_p_idx = int(pidxs[best_p_i])
        best_p_time = float(best_p_idx * sf)
        best_p_prob = float(pinfo["peak_heights"][best_p_i])
    else:
        best_p_idx = -1
        best_p_time = -1.0
        best_p_prob = -1.0

    if len(sidxs) > 0:
        best_s_i = int(np.argmax(sinfo["peak_heights"]))
        best_s_idx = int(sidxs[best_s_i])
        best_s_time = float(best_s_idx * sf)
        best_s_prob = float(sinfo["peak_heights"][best_s_i])
    else:
        best_s_idx = -1
        best_s_time = -1.0
        best_s_prob = -1.0

    return {
        "num_p_peaks": int(len(pidxs)),
        "num_s_peaks": int(len(sidxs)),
        "best_p_idx": best_p_idx,
        "best_p_time_sec": best_p_time,
        "best_p_prob": best_p_prob,
        "best_s_idx": best_s_idx,
        "best_s_time_sec": best_s_time,
        "best_s_prob": best_s_prob,
        "max_p_prob": float(np.max(pred[0, :])),
        "max_s_prob": float(np.max(pred[1, :])),
        "max_bg_prob": float(np.max(pred[2, :])),
    }


def evaluate_pick(pred_idx, gt_idx, tol):
    pred_exists = pred_idx >= 0
    gt_exists = gt_idx >= 0

    if gt_exists and pred_exists:
        err = int(pred_idx - gt_idx)
        if abs(err) <= tol:
            return {
                "tp": 1, "fp": 0, "fn": 0, "tn": 0,
                "matched": 0,
                "error_samples": err,
                "abs_error_samples": abs(err),
            }
        else:
            return {
                "tp": 0, "fp": 1, "fn": 0, "tn": 0,
                "matched": 0,
                "error_samples": err,
                "abs_error_samples": abs(err),
            }

    if gt_exists and not pred_exists:
        return {
            "tp": 0, "fp": 0, "fn": 1, "tn": 0,
            "matched": 0,
            "error_samples": None,
            "abs_error_samples": None,
        }

    if (not gt_exists) and pred_exists:
        return {
            "tp": 0, "fp": 1, "fn": 0, "tn": 0,
            "matched": 1,
            "error_samples": None,
            "abs_error_samples": None,
        }

    return {
        "tp": 0, "fp": 0, "fn": 0, "tn": 1,
        "matched": 0,
        "error_samples": None,
        "abs_error_samples": None,
    }

def aggregate_phase_metrics(eval_rows):
    tp = sum(r["tp"] for r in eval_rows)
    fp = sum(r["fp"] for r in eval_rows)
    fn = sum(r["fn"] for r in eval_rows)
    tn = sum(r["tn"] for r in eval_rows)

    precision, recall, acc, fscore = calc_metrics(tp, fp, fn, tn)

    abs_errs = [
        r["abs_error_samples"]
        for r in eval_rows
        if r["abs_error_samples"] is not None and r["tp"] == 1
    ]

    signed_errs = [
        r["error_samples"]
        for r in eval_rows
        if r["error_samples"] is not None and r["tp"] == 1
    ]

    mean_abs_error_samples = float(np.mean(abs_errs)) if len(abs_errs) > 0 else -1.0
    median_abs_error_samples = float(np.median(abs_errs)) if len(abs_errs) > 0 else -1.0

    mean_signed_error_samples = float(np.mean(signed_errs)) if len(signed_errs) > 0 else -1.0
    std_signed_error_samples = float(np.std(signed_errs)) if len(signed_errs) > 0 else -1.0

    mean_abs_error_sec = mean_abs_error_samples / 100.0 if mean_abs_error_samples >= 0 else -1.0
    median_abs_error_sec = median_abs_error_samples / 100.0 if median_abs_error_samples >= 0 else -1.0

    mean_signed_error_sec = mean_signed_error_samples / 100.0 if mean_signed_error_samples >= 0 else -1.0
    std_signed_error_sec = std_signed_error_samples / 100.0 if std_signed_error_samples >= 0 else -1.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "f1": float(fscore),

        "matched_count": int(sum(r["matched"] for r in eval_rows)),

        "mean_abs_error_samples": mean_abs_error_samples,
        "median_abs_error_samples": median_abs_error_samples,
        "mean_signed_error_samples": mean_signed_error_samples,
        "std_signed_error_samples": std_signed_error_samples,

        "mean_abs_error_sec": mean_abs_error_sec,
        "median_abs_error_sec": median_abs_error_sec,
        "mean_signed_error_sec": mean_signed_error_sec,
        "std_signed_error_sec": std_signed_error_sec,
    }


def plot_result_3C(wave, pred, station, trace_name, sf,
                   p_gt_idx=None, s_gt_idx=None,
                   save_path=None, show_fig=False):
    lapse_time = np.arange(0, 30, sf)

    summary = summarize_prediction(pred, sf=sf)
    best_p_idx = summary["best_p_idx"]
    best_s_idx = summary["best_s_idx"]

    z_filt = bandpass_filter(wave[0, :], fs=1.0 / sf, fmin=1.0, fmax=30.0, order=4)

    fig = plt.figure(figsize=(10, 6.5))

    ax1 = fig.add_subplot(5, 1, 1)
    plot_wave(ax1, lapse_time, wave[0, :], best_p_idx, best_s_idx, sf, "Z",
              p_gt_idx=p_gt_idx, s_gt_idx=s_gt_idx)

    ax2 = fig.add_subplot(5, 1, 2)
    plot_wave(ax2, lapse_time, wave[1, :], best_p_idx, best_s_idx, sf, "N",
              p_gt_idx=p_gt_idx, s_gt_idx=s_gt_idx)

    ax3 = fig.add_subplot(5, 1, 3)
    plot_wave(ax3, lapse_time, wave[2, :], best_p_idx, best_s_idx, sf, "E",
              p_gt_idx=p_gt_idx, s_gt_idx=s_gt_idx)

    ax4 = fig.add_subplot(5, 1, 4)
    plot_wave(ax4, lapse_time, z_filt, best_p_idx, best_s_idx, sf, "Z 1-30Hz",
              p_gt_idx=p_gt_idx, s_gt_idx=s_gt_idx)

    ax5 = fig.add_subplot(5, 1, 5)
    plot_pred(ax5, lapse_time, pred[0, :], best_p_idx, "P", sf)
    plot_pred(ax5, lapse_time, pred[1, :], best_s_idx, "S", sf)

    if p_gt_idx is not None and p_gt_idx >= 0:
        ax5.axvline(x=p_gt_idx * sf, c="green", lw=1.2, linestyle="--", label="P_GT_sample")

    if s_gt_idx is not None and s_gt_idx >= 0:
        ax5.axvline(x=s_gt_idx * sf, c="orange", lw=1.2, linestyle="--", label="S_GT_sample")

    ax5.legend()
    ax1.set_title(f"{station} | {trace_name}", fontsize=14)
    plt.subplots_adjust(hspace=0.2)

    ax1.yaxis.set_label_coords(-0.06, 0.5)
    ax2.yaxis.set_label_coords(-0.06, 0.5)
    ax3.yaxis.set_label_coords(-0.06, 0.5)
    ax4.yaxis.set_label_coords(-0.06, 0.5)
    ax5.yaxis.set_label_coords(-0.06, 0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_fig:
        plt.show()

    plt.close(fig)

import shutil

def main():
    if os.path.exists(OUTPUT_DIR):
        print("[DEBUG] Removing old OUTPUT_DIR:", OUTPUT_DIR)
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    fp_noise_dir = os.path.join(OUTPUT_DIR, "fp_noise")
    fp_wrong_dir = os.path.join(OUTPUT_DIR, "fp_wrong_pick")
    
    os.makedirs(fp_noise_dir, exist_ok=True)
    os.makedirs(fp_wrong_dir, exist_ok=True)
    
    fp_noise_count = 0
    fp_wrong_count = 0
    MAX_FP_NOISE_FIGS = 5
    MAX_FP_WRONG_FIGS = 5
    os.makedirs(fig_dir, exist_ok=True)

    meta_df = load_metadata_csv(META_CSV_PATH)
    pred_meta_df = build_prediction_meta_df(
        meta_df=meta_df,
        pred_source_mode=PRED_SOURCE_MODE,
        target_csv_path=TARGET_CSV_PATH,
    )

    print(f"PRED_SOURCE_MODE = {PRED_SOURCE_MODE}")
    print(f"Prediction rows = {len(pred_meta_df)}")
    if PRED_SOURCE_MODE == "csv_list":
        print(f"TARGET_CSV_PATH = {TARGET_CSV_PATH}")
        print("NOTE: split column is ignored completely.")


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
    elif method == "FUSION_LST_SST":
        from model_sstnet import ModelLST, ModelSST,FixedSTransformTarget,StandardSTransformTarget
        sys.path.pop()
    
        model_lst = ModelLST(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    
        model_sst = ModelSST(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
        target_generator = FixedSTransformTarget(
            in_channels=3,
            sample_rate=SST_SAMPLE_RATE,
            n_freq=SST_N_FREQ,
            kernel_size=SST_KERNEL_SIZE,
            f_min=0.5,
            f_max=45.0,
            topk_ratio=SST_TOPK_RATIO,
            use_log_compression=True,
        ).to(device)
        '''target_generator = StandardSTransformTarget(
            in_channels=3,
            sample_rate=100,
            n_freq=64,
            kernel_size=129,
            f_min=0.5,
            f_max=45.0,
            use_log_compression=True, 
            merge_components=True,
        ).to(device)'''
        target_generator.eval()
        print(f"Loading LST model from: {LST_MODEL_PATH}")
        model_lst.load_state_dict(torch.load(LST_MODEL_PATH, map_location=device), strict=True)
    
        print(f"Loading SST model from: {SST_MODEL_PATH}")
        model_sst.load_state_dict(torch.load(SST_MODEL_PATH, map_location=device), strict=True)
    
        model_lst.eval()
        model_sst.eval()
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
    elif method=='LFB':
        from model_learnableFilter import ModelLFB
        from model_sstnet import ModelLST, ModelSST,FixedSTransformTarget,StandardSTransformTarget
        sys.path.pop()
        model = ModelLFB(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
        target_generator = FixedSTransformTarget(
            in_channels=3,
            sample_rate=SST_SAMPLE_RATE,
            n_freq=SST_N_FREQ,
            kernel_size=SST_KERNEL_SIZE,
            f_min=0.5,
            f_max=45.0,
            topk_ratio=SST_TOPK_RATIO,
            use_log_compression=True,
        ).to(device)
    
        target_generator.eval()
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
    elif method=='SST':
        from model_sstnet import ModelSST, FixedSTransformTarget, SSTNetLoss
        sys.path.pop()
    
        model = ModelSST(
            in_length=100 * 30,
            in_channels=3,
            class_num=3,
            strides=[3, 2, 2],
            kernel_size=3,
        ).to(device)
    
        target_generator = FixedSTransformTarget(
            in_channels=3,
            sample_rate=SST_SAMPLE_RATE,
            n_freq=SST_N_FREQ,
            kernel_size=SST_KERNEL_SIZE,
            f_min=0.5,
            f_max=45.0,
            topk_ratio=SST_TOPK_RATIO,
            use_log_compression=True,
        ).to(device)
    
        sst_loss_fn = SSTNetLoss(
            l1_weight=1.0,
            sparse_weight=1e-4,
            smooth_weight=1e-4,
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
    if method!='FUSION_LST_SST':
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        '''total_params, trainable_params = count_parameters(model, method)
    
        macs, thop_params = profile_macs(
            model,
            device,
            input_shape=(1, 3, 3000),
            name=method
        )
    
        infer_ms, traces_per_sec = benchmark_inference_time(
            model,
            device,
            input_shape=(1, 3, 3000),
            warmup=30,
            runs=100,
            name=method
        )
    
        peak_allocated_mb, peak_reserved_mb = benchmark_inference_memory(
            model,
            device,
            input_shape=(1, 3, 3000),
            name=method
        )   
        peak_delta_gpu_mem_mb, peak_total_gpu_mem_mb, baseline_gpu_mem_mb = benchmark_inference_memory_nvidia_smi_delta(
            model,
            device,
            input_shape=(1, 3, 3000),
            gpuid=None,
            runs=30,
            name=method
        )
        print(
            "method", method,
            "total_params", int(total_params),
            "trainable_params", int(trainable_params),
            "macs", macs,
            "inference_ms_per_trace", float(infer_ms),
            "throughput_traces_per_sec", float(traces_per_sec),
            "peak_infer_allocated_mb", float(peak_allocated_mb),
            "peak_infer_reserved_mb", float(peak_reserved_mb),
            "peak_infer_delta_gpu_mem_mb", float(peak_delta_gpu_mem_mb),
            "peak_infer_total_gpu_mem_mb", float(peak_total_gpu_mem_mb),
            "baseline_gpu_mem_mb", float(baseline_gpu_mem_mb)
        )
    else:
        lst_total, lst_trainable, _ = count_parameters(model_lst, "ModelLST")
        sst_total, sst_trainable, _ = count_parameters(model_sst, "ModelSST")
        
        print("\n=== Fusion Total ===")
        print(f"Total parameters:     {lst_total + sst_total:,}")
        print(f"Trainable parameters: {lst_trainable + sst_trainable:,}")'''
    csv_rows = []
    p_eval_rows = []
    s_eval_rows = []
    event_eval_rows = []
    nproc = 0

    '''data_iter = iter_raw_h5_csv_for_pred(
        H5_PATH,
        meta_df=pred_meta_df,
        window_mode=WINDOW_MODE,
        start_sample=CUSTOM_START_SAMPLE,
    )'''
    data_iter, skip_counter = iter_raw_h5_csv_for_pred(
        H5_PATH,
        meta_df=pred_meta_df,
        window_mode=WINDOW_MODE,
        start_sample=CUSTOM_START_SAMPLE,
        return_skip_counter=True,
    )
    for i, (station, trace_name, wave, meta) in enumerate(tqdm(data_iter, desc="Processing predictions")):
        if MAX_PLOTS is not None and i >= MAX_PLOTS:
            break

        x = torch.from_numpy(wave).unsqueeze(0).float().to(device)  # (1, 3, 3000)
        with torch.no_grad():
            pred_spec = None
            lst_spec = None
            '''if method == "FUSION_LST_SST":
                out_lst, lst_spec = model_lst(x, return_tfmap=True)
            
                out_sst, pred_spec = model_sst(x, return_sst=True)
            
                if isinstance(out_lst, tuple):
                    out_lst = out_lst[0]
            
                pred_lst = out_lst.detach().cpu().numpy()[0]
                pred_sst = out_sst.detach().cpu().numpy()[0]
            
                pred = fuse_lst_sst(
                    pred_lst,
                    pred_sst,
                    noise_threshold=0.8
                )
            elif method == "SST":
                model_out, pred_spec = model(x, return_sst=True)
                pred = model_out.detach().cpu().numpy()[0]
            elif method == "LST":
                model_out, lst_spec = model(x,ablation=ablation, return_tfmap=True)
            
                if isinstance(model_out, tuple):
                    model_out = model_out[0]
            
                pred = model_out.detach().cpu().numpy()[0]
            else:'''
            model_out = model(x)
    
            if isinstance(model_out, tuple):
                model_out = model_out[0]
    
            pred = model_out.detach().cpu().numpy()[0]
        summary = summarize_prediction(pred, sf=0.01)
        summary["station"] = station
        summary["trace_name"] = trace_name
        summary.update(meta)
        if i < 5 and  (pred_spec is not None or lst_spec is not None):
            (wave_np,wave_norm01,wave_bp145,stft_spec,sparse_spec,lst_plot_spec,sstnet_spec,) = make_tf_specs_for_plot(x,pred_spec=pred_spec,target_generator=target_generator,lst_spec=lst_spec )
            save_path = os.path.join( fig_dir, f"{i:04d}_{safe_name(trace_name)}_wave_stft_sparse_sst.png")
            plot_wave_tf_stack(
                wave=wave_np,
                wave_norm01=wave_norm01,
                wave_bp145=wave_bp145,
                stft_spec=stft_spec,
                sparse_spec=sparse_spec,
                lst_spec=lst_plot_spec,
                sst_spec=sstnet_spec,
                title=f"{station} | {trace_name}",
                save_path=save_path,
                p_gt_idx=summary["p_arrival_sample_in_window"],
                s_gt_idx=summary["s_arrival_sample_in_window"],
                p_pick_idx=summary["best_p_idx"],
                s_pick_idx=summary["best_s_idx"],
            )
        p_eval = evaluate_pick(
            pred_idx=summary["best_p_idx"],
            gt_idx=summary["p_arrival_sample_in_window"],
            tol=TOLERANCE_SAMPLES
        )
        s_eval = evaluate_pick(
            pred_idx=summary["best_s_idx"],
            gt_idx=summary["s_arrival_sample_in_window"],
            tol=TOLERANCE_SAMPLES
        )
        # =========================================================
        # Event detection evaluation
        #   GT event:
        #       trace_category not noise or P/S GT in window 
        #   Pred event:
        #       model pred. P or S peak
        # =========================================================
        
        trace_category = str(summary.get("trace_category", "")).lower()
        
        if "noise" in trace_category:
            gt_event = False
        elif trace_category != "":
            gt_event = True
        else:
            gt_event = (
                summary["p_arrival_sample_in_window"] >= 0
                or summary["s_arrival_sample_in_window"] >= 0
            )
        
        pred_event = (
            summary["best_p_idx"] >= 0
            or summary["best_s_idx"] >= 0
        )
        
        event_eval = evaluate_event_detection(
            pred_event=pred_event,
            gt_event=gt_event
        )
        
        summary["event_gt"] = int(gt_event)
        summary["event_pred"] = int(pred_event)
        summary["event_tp"] = event_eval["tp"]
        summary["event_fp"] = event_eval["fp"]
        summary["event_fn"] = event_eval["fn"]
        summary["event_tn"] = event_eval["tn"]
        summary["p_tp"] = p_eval["tp"]
        summary["p_fp"] = p_eval["fp"]
        summary["p_fn"] = p_eval["fn"]
        summary["p_tn"] = p_eval["tn"]
        summary["p_matched"] = p_eval["matched"]
        summary["p_error_samples"] = p_eval["error_samples"] if p_eval["error_samples"] is not None else ""
        summary["p_abs_error_samples"] = p_eval["abs_error_samples"] if p_eval["abs_error_samples"] is not None else ""

        summary["s_tp"] = s_eval["tp"]
        summary["s_fp"] = s_eval["fp"]
        summary["s_fn"] = s_eval["fn"]
        summary["s_tn"] = s_eval["tn"]
        summary["s_matched"] = s_eval["matched"]
        summary["s_error_samples"] = s_eval["error_samples"] if s_eval["error_samples"] is not None else ""
        summary["s_abs_error_samples"] = s_eval["abs_error_samples"] if s_eval["abs_error_samples"] is not None else ""
                # =========================================================
        # Save FP diagnostic figures
        #   noise FP:
        #       GT does not exist, but model predicts a pick.
        #   wrong-pick FP:
        #       GT exists and model predicts a pick, but error > tolerance.
        # =========================================================
        if fp_noise_count < MAX_FP_NOISE_FIGS or fp_wrong_count < MAX_FP_WRONG_FIGS:
            if pred_spec is not None and target_generator is not None:
                p_gt = summary["p_arrival_sample_in_window"]
                s_gt = summary["s_arrival_sample_in_window"]
                p_pick = summary["best_p_idx"]
                s_pick = summary["best_s_idx"]
    
                # P noise FP: no P GT, but P pick exists
                p_noise_fp = (p_gt < 0 and p_pick >= 0)
    
                # S noise FP: no S GT, but S pick exists
                s_noise_fp = (s_gt < 0 and s_pick >= 0)
    
                # P wrong-pick FP: P GT exists, P pick exists, but error > tolerance
                p_wrong_fp = (
                    p_gt >= 0
                    and p_pick >= 0
                    and abs(p_pick - p_gt) > TOLERANCE_SAMPLES
                )
    
                # S wrong-pick FP: S GT exists, S pick exists, but error > tolerance
                s_wrong_fp = (
                    s_gt >= 0
                    and s_pick >= 0
                    and abs(s_pick - s_gt) > TOLERANCE_SAMPLES
                )
    
                if (
                    (p_noise_fp or s_noise_fp)
                    and fp_noise_count < MAX_FP_NOISE_FIGS
                ):
                    (wave_np,wave_norm01,wave_bp145,stft_spec,sparse_spec,lst_plot_spec,sstnet_spec,) = make_tf_specs_for_plot(x,pred_spec=pred_spec,target_generator=target_generator,lst_spec=lst_spec )
                    phase_tag = []
                    if p_noise_fp:
                        phase_tag.append("P")
                    if s_noise_fp:
                        phase_tag.append("S")
                    phase_tag = "".join(phase_tag)
    
                    save_path = os.path.join(
                        fp_noise_dir,
                        f"{fp_noise_count:02d}_noiseFP_{phase_tag}_{safe_name(trace_name)}.png"
                    )
    
                    plot_wave_tf_stack(
                          wave=wave_np,
                          wave_norm01=wave_norm01,
                          wave_bp145=wave_bp145,
                          stft_spec=stft_spec,
                          sparse_spec=sparse_spec,
                          lst_spec=lst_plot_spec,
                          sst_spec=sstnet_spec,
                          title=f"{station} | {trace_name}",
                          save_path=save_path,
                          p_gt_idx=summary["p_arrival_sample_in_window"],
                          s_gt_idx=summary["s_arrival_sample_in_window"],
                          p_pick_idx=summary["best_p_idx"],
                          s_pick_idx=summary["best_s_idx"],
                    )
    
                    fp_noise_count += 1
    
                if (
                    (p_wrong_fp or s_wrong_fp)
                    and fp_wrong_count < MAX_FP_WRONG_FIGS
                ):
                    
                    (wave_np,wave_norm01,wave_bp145,stft_spec,sparse_spec,lst_plot_spec,sstnet_spec,) = make_tf_specs_for_plot(x,pred_spec=pred_spec,target_generator=target_generator,lst_spec=lst_spec )
                    phase_tag = []
                    if p_wrong_fp:
                        phase_tag.append("P")
                    if s_wrong_fp:
                        phase_tag.append("S")
                    phase_tag = "".join(phase_tag)
    
                    save_path = os.path.join(
                        fp_wrong_dir,
                        f"{fp_wrong_count:02d}_wrongPickFP_{phase_tag}_{safe_name(trace_name)}.png"
                    )
                    plot_wave_tf_stack(
                        wave=wave_np,
                        wave_norm01=wave_norm01,
                        wave_bp145=wave_bp145,
                        stft_spec=stft_spec,
                        sparse_spec=sparse_spec,
                        lst_spec=lst_plot_spec,
                        sst_spec=sstnet_spec,
                        title=f"{station} | {trace_name}",
                        save_path=save_path,
                        p_gt_idx=summary["p_arrival_sample_in_window"],
                        s_gt_idx=summary["s_arrival_sample_in_window"],
                        p_pick_idx=summary["best_p_idx"],
                        s_pick_idx=summary["best_s_idx"],
                    )
    
                    fp_wrong_count += 1
        if SAVE_FIG:
            fig_name = f"{i:07d}_{safe_name(station)}_{safe_name(trace_name)}.png"
            fig_path = os.path.join(fig_dir, fig_name)
            plot_result_3C(
                wave, pred, station, trace_name, 0.01,
                p_gt_idx=meta["p_arrival_sample_in_window"],
                s_gt_idx=meta["s_arrival_sample_in_window"],
                save_path=fig_path,
                show_fig=SHOW_FIG
            )
            summary["figure_path"] = fig_path
        else:
            summary["figure_path"] = ""
        event_eval_rows.append(event_eval)
        csv_rows.append(summary)
        p_eval_rows.append(p_eval)
        s_eval_rows.append(s_eval)
        nproc += 1
    
    print("\n=== Load / Skip Summary ===")
    print("Prediction rows:", len(pred_meta_df))
    for k, v in skip_counter.items():
        print(f"{k}: {v}")
    print("skipped_total:", len(pred_meta_df) - skip_counter.get("processed", 0))
    
    if nproc == 0:
        print("No valid traces found from HDF5 + selected CSV source.")
        return
    if SAVE_CSV:
        csv_path = os.path.join(OUTPUT_DIR, "prediction_summary.csv")
        fieldnames = [
            "station", "trace_name",
            "window_start_sample", "window_end_sample",
            "event_gt", "event_pred",
            "event_tp", "event_fp", "event_fn", "event_tn",
            "p_arrival_sample", "s_arrival_sample",
            "p_arrival_sample_in_window", "s_arrival_sample_in_window",
            "trace_category", "trace_snr_db", "trace_p_status", "trace_s_status",
            "num_p_peaks", "num_s_peaks",
            "best_p_idx", "best_p_time_sec", "best_p_prob",
            "best_s_idx", "best_s_time_sec", "best_s_prob",
            "max_p_prob", "max_s_prob", "max_bg_prob",
            "p_tp", "p_fp", "p_fn", "p_tn", "p_matched", "p_error_samples", "p_abs_error_samples",
            "s_tp", "s_fp", "s_fn", "s_tn", "s_matched", "s_error_samples", "s_abs_error_samples",
            "figure_path"
        ]
    
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print("Saved CSV:", csv_path)
    
        metrics_summary = {
            "pred_source_mode": PRED_SOURCE_MODE,
            "data_source_h5": H5_PATH,
            "data_source_csv": META_CSV_PATH,
            "target_csv_path": TARGET_CSV_PATH if PRED_SOURCE_MODE == "csv_list" else "",
            "tolerance_samples": int(TOLERANCE_SAMPLES),
            "tolerance_sec": float(TOLERANCE_SAMPLES / 100.0),
            "num_processed_traces": int(nproc),
            "Event_Detection": aggregate_detection_metrics(event_eval_rows),
            "P": aggregate_phase_metrics(p_eval_rows),
            "S": aggregate_phase_metrics(s_eval_rows),
        }
    
        metrics_json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print("Saved metrics JSON:", metrics_json_path)
    
        metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    
        metric_fieldnames = [
            "phase", "tp", "fp", "fn", "tn",
            "precision", "recall", "accuracy", "f1",
            "matched_count",
            "mean_abs_error_samples",
            "median_abs_error_samples",
            "mean_signed_error_samples",
            "std_signed_error_samples",
            "mean_abs_error_sec",
            "median_abs_error_sec",
            "mean_signed_error_sec",
            "std_signed_error_sec",
        ]
    
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
            writer.writeheader()
    
            for phase in ["Event_Detection", "P", "S"]:
                row = {"phase": phase}
                row.update(metrics_summary[phase])
    
                row = {k: row.get(k, "") for k in metric_fieldnames}
    
                writer.writerow(row)
    
        print("Saved metrics CSV:", metrics_csv_path)
    
        print("\n=== Metrics Summary ===")
        for phase in ["Event_Detection", "P", "S"]:
            m = metrics_summary[phase]
    
            msg = (
                f"{phase}: "
                f"precision={m['precision']:.4f}, "
                f"recall={m['recall']:.4f}, "
                f"accuracy={m['accuracy']:.4f}, "
                f"f1={m['f1']:.4f}, "
                f"tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']}"
            )
    
            if phase in ["P", "S"]:
                msg += (
                    f", matched_count={m['matched_count']}, "
                    f"MAE={m['mean_abs_error_samples']:.2f} samples, "
                    f"Median_MAE={m['median_abs_error_samples']:.2f} samples, "
                    f"Residual_mean={m['mean_signed_error_samples']:.2f} samples, "
                    f"Residual_std={m['std_signed_error_samples']:.2f} samples"
                )
    
            print(msg)
    
    print("Done. Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    device = torch.device(DEVICE)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("DEVICE =", DEVICE)
    print("device =", device)
    main()