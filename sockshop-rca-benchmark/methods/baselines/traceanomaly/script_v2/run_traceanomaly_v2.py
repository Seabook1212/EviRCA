import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str((SCRIPT_DIR / ".mplconfig").resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, TensorDataset


TELEMETRY_DAYS = ["2026_03_12", "2026_03_13", "2026_03_14", "2026_03_17", "2026_03_18"]
EXP_ID = None

FAULT_WINDOW_MINUTES = 5
NORMAL_WINDOW_OFFSET_MINUTES = 0
NORMAL_WINDOW_DURATION_MINUTES = None

ANOMALOUS_STD_THRESHOLD = 3.0
P_VALUE_ALPHA = 0.001
MIN_VALID_TRACE_DIMS = 1
MIN_HSTV_REFERENCE_TRACES = 1
IMPORTANCE_SAMPLES = 32
FLOW_LAYERS = 0
FLOW_HIDDEN_DIM = 32
KDE_GRID_SIZE = 2048

HIDDEN_DIM = 500
LATENT_DIM = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 2000
VALIDATION_SPLIT = 0.1
WEIGHT_DECAY = 1e-4
LR_ANNEAL_FACTOR = 0.5
LR_ANNEAL_EPOCH_FREQ = 100
GRAD_NORM_CLIP = 10.0
EARLY_STOPPING_PATIENCE_EVALS = 10
RANDOM_SEED = 20260320

SERVICE_RANKING_MODE = "leaf_only"
SERVICE_RANKING_SORT_MODE = "count_then_score"
SERVICE_PATH_ANCESTOR_WEIGHT = 0.4
USE_SECONDARY_ANOMALOUS_DIMENSIONS = True
SECONDARY_DIMENSION_WEIGHT = 0.5
MAX_SECONDARY_DIMENSIONS_PER_TRACE = 5
TOPK_FILL_ENABLED = True
TOPK_FILL_K = 5
TOPK_FILL_MODE = "path_ancestors_only"

OUTPUT_ROOT = "../data_v2"
NORMAL_BASELINE_DIR = "normal_baseline"
MODEL_DIR = "model"
CALL_PATH_FILE = "call_path_list.json"
NORMALIZATION_FILE = "normalization_stats.json"
TRAINING_LOSS_FILE = "training_loss.json"
NORMAL_BASELINE_SUMMARY_FILE = "normal_baseline_summary.json"
TRACE_SCORES_FILE = "trace_scores.csv"
TRACE_LOCALIZATION_FILE = "trace_root_causes.csv"
SERVICE_RANKING_FILE = "service_ranking.csv"
RUN_SUMMARY_FILE = "traceanomaly_run_summary.json"
DAY_DETAILS_FILE = "traceanomaly_accuracy_details.csv"
DAY_SUMMARY_FILE = "traceanomaly_accuracy_summary.csv"
UNCLEAR_POINTS_FILE = "traceanomaly_unclear_points.json"
ALL_DAYS_DETAILS_FILE = "traceanomaly_accuracy_details_all_days.csv"
ALL_DAYS_SUMMARY_FILE = "traceanomaly_accuracy_summary_all_days.csv"
OVERALL_SUMMARY_FILE = "traceanomaly_accuracy_summary_overall.csv"

FAULT_RUN_ROOT_TEMPLATE = "../../../dataset_v2/fault_run_{day_suffix}"
NORMAL_RUN_ROOT_TEMPLATE = "../../../dataset_v2/normal_run_{day_suffix}"
TELEMETRY_TRACES_DIR_TEMPLATE = "../../../dataset_v2/telemetry/{telemetry_day}/traces"

TRACE_USECOLS = [
    "timestamp",
    "trace_id",
    "span_id",
    "parent_span_id",
    "service",
    "duration",
    "span_kind",
]
LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class RunWindow:
    run_id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    metadata_path: Path
    inject_start: pd.Timestamp | None = None
    ground_truth_service: str | None = None


@dataclass
class KdeTailModel:
    values: np.ndarray
    grid: np.ndarray
    cdf: np.ndarray
    bandwidth: float


class AffineCouplingFlow(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, hidden_dim: int, flip: bool):
        super().__init__()
        half_dim = latent_dim // 2
        self.half_dim = half_dim
        self.flip = flip
        self.net = nn.Sequential(
            nn.Linear(half_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, half_dim * 2),
        )

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            z1, z2 = z[:, self.half_dim :], z[:, : self.half_dim]
        else:
            z1, z2 = z[:, : self.half_dim], z[:, self.half_dim :]

        params = self.net(torch.cat([z1, context], dim=1))
        shift, log_scale = params.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)
        transformed = z2 * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=1)

        if self.flip:
            out = torch.cat([transformed, z1], dim=1)
        else:
            out = torch.cat([z1, transformed], dim=1)
        return out, log_det


class PosteriorFlowVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, flow_layers: int, flow_hidden_dim: int):
        super().__init__()
        if flow_layers > 0 and latent_dim % 2 != 0:
            raise ValueError("LATENT_DIM must be even for affine coupling flow.")

        context_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(context_dim, latent_dim)
        self.fc_logvar = nn.Linear(context_dim, latent_dim)
        self.flows = nn.ModuleList(
            [
                AffineCouplingFlow(
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    hidden_dim=flow_hidden_dim,
                    flip=bool(index % 2),
                )
                for index in range(flow_layers)
            ]
        )
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, context_dim),
            nn.LeakyReLU(),
            nn.Linear(context_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_x_mean = nn.Linear(hidden_dim, input_dim)
        self.fc_x_logvar = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encoder(x)
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        return context, mu, logvar

    def _log_normal(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (LOG_2PI + logvar + ((x - mu) ** 2) / torch.exp(logvar))

    def _log_standard_normal(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (LOG_2PI + x.pow(2))

    def sample_posterior(
        self,
        context: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        log_q0 = self._log_normal(z0, mu, logvar).sum(dim=1)

        zk = z0
        sum_log_det = torch.zeros(z0.size(0), device=z0.device)
        for flow in self.flows:
            zk, log_det = flow(zk, context)
            sum_log_det = sum_log_det + log_det
        log_qk = log_q0 - sum_log_det
        return z0, zk, log_qk

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.decoder_hidden(z)
        x_mean = self.fc_x_mean(hidden)
        x_logvar = self.fc_x_logvar(hidden)
        x_logvar = torch.log(1e-4 + torch.nn.functional.softplus(x_logvar))
        x_logvar = torch.clamp(x_logvar, min=-8.0, max=8.0)
        return x_mean, x_logvar

    def _log_px_given_z(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        recon_mean: torch.Tensor,
        recon_logvar: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * (LOG_2PI + recon_logvar + ((x - recon_mean) ** 2) / torch.exp(recon_logvar)) * mask

    def elbo(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context, mu, logvar = self.encode(x)
        _, zk, log_qk = self.sample_posterior(context, mu, logvar)
        recon_mean, recon_logvar = self.decode(zk)
        log_px = self._log_px_given_z(x, mask, recon_mean, recon_logvar).sum(dim=1)
        log_pz = self._log_standard_normal(zk).sum(dim=1)
        elbo = log_px + log_pz - log_qk
        return elbo, recon_mean

    def estimate_log_likelihood(self, x: torch.Tensor, mask: torch.Tensor, num_samples: int) -> torch.Tensor:
        context, mu, logvar = self.encode(x)
        log_weights = []
        for _ in range(num_samples):
            _, zk, log_qk = self.sample_posterior(context, mu, logvar)
            recon_mean, recon_logvar = self.decode(zk)
            log_px = self._log_px_given_z(x, mask, recon_mean, recon_logvar).sum(dim=1)
            log_pz = self._log_standard_normal(zk).sum(dim=1)
            log_weights.append(log_px + log_pz - log_qk)
        stacked = torch.stack(log_weights, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(num_samples)


class HourlyTraceCache:
    def __init__(self):
        self._cache: dict[Path, pd.DataFrame] = {}

    def load_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(file_path, usecols=TRACE_USECOLS, chunksize=200_000):
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk["duration"] = pd.to_numeric(chunk["duration"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp", "trace_id", "span_id", "service", "duration"])
            if chunk.empty:
                continue
            chunk["timestamp"] = chunk["timestamp"].astype(np.int64)
            chunk["service"] = chunk["service"].astype(str)
            chunk["span_kind"] = chunk["span_kind"].fillna("").astype(str)
            chunks.append(chunk)

        if chunks:
            loaded = pd.concat(chunks, ignore_index=True)
        else:
            loaded = pd.DataFrame(columns=TRACE_USECOLS)
        self._cache[file_path] = loaded
        return loaded


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.tz_convert(None)


def timestamp_to_us(value: pd.Timestamp) -> int:
    return int(value.value // 1_000)


def telemetry_day_to_suffix(telemetry_day: str) -> str:
    return datetime.strptime(telemetry_day, "%Y_%m_%d").strftime("%m%d")


def get_day_paths(script_dir: Path, telemetry_day: str) -> tuple[Path, Path, Path]:
    day_suffix = telemetry_day_to_suffix(telemetry_day)
    fault_root = (script_dir / FAULT_RUN_ROOT_TEMPLATE.format(day_suffix=day_suffix)).resolve()
    normal_root = (script_dir / NORMAL_RUN_ROOT_TEMPLATE.format(day_suffix=day_suffix)).resolve()
    traces_dir = (script_dir / TELEMETRY_TRACES_DIR_TEMPLATE.format(telemetry_day=telemetry_day)).resolve()
    return fault_root, normal_root, traces_dir


def discover_fault_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    fault_root, _, _ = get_day_paths(script_dir, telemetry_day)
    runs: list[RunWindow] = []
    for exp_dir in sorted(fault_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        metadata_path = exp_dir / "fault_info" / "fault_metadata.json"
        workload_path = exp_dir / "workload" / "workload_metadata.json"
        if not metadata_path.exists():
            continue

        metadata = _read_json(metadata_path)
        injection_info = metadata.get("injection_info", {})
        inject_start_raw = injection_info.get("inject_start")
        ground_truth_service = injection_info.get("service") or metadata.get("service")
        if not inject_start_raw or not ground_truth_service:
            continue

        inject_start = parse_utc_timestamp(inject_start_raw)
        start = inject_start
        end = inject_start + pd.Timedelta(minutes=FAULT_WINDOW_MINUTES)
        if workload_path.exists():
            workload = _read_json(workload_path)
            workload_start_raw = workload.get("workload_start_time")
            workload_end_raw = workload.get("workload_end_time")
            if workload_start_raw and workload_end_raw:
                start = parse_utc_timestamp(workload_start_raw)
                end = parse_utc_timestamp(workload_end_raw)

        if end <= start:
            continue

        runs.append(
            RunWindow(
                run_id=exp_dir.name,
                label="fail",
                start=start,
                end=end,
                metadata_path=metadata_path.resolve(),
                inject_start=inject_start,
                ground_truth_service=str(ground_truth_service),
            )
        )

    if not runs:
        raise FileNotFoundError(f"No fault runs found for {telemetry_day}")
    return runs


def discover_normal_runs(script_dir: Path, telemetry_day: str) -> list[RunWindow]:
    _, normal_root, _ = get_day_paths(script_dir, telemetry_day)
    runs: list[RunWindow] = []
    for normal_dir in sorted(normal_root.iterdir()):
        if not normal_dir.is_dir():
            continue
        metadata_path = normal_dir / "workload" / "workload_metadata.json"
        if not metadata_path.exists():
            continue

        metadata = _read_json(metadata_path)
        workload_start_raw = metadata.get("workload_start_time")
        workload_end_raw = metadata.get("workload_end_time")
        if not workload_start_raw or not workload_end_raw:
            continue

        workload_start = parse_utc_timestamp(workload_start_raw)
        workload_end = parse_utc_timestamp(workload_end_raw)
        normal_start = workload_start + pd.Timedelta(minutes=NORMAL_WINDOW_OFFSET_MINUTES)
        if NORMAL_WINDOW_DURATION_MINUTES is None:
            normal_end = workload_end
        else:
            normal_end = normal_start + pd.Timedelta(minutes=NORMAL_WINDOW_DURATION_MINUTES)
        if normal_end > workload_end:
            normal_end = workload_end
        if normal_end <= normal_start:
            continue

        runs.append(
            RunWindow(
                run_id=normal_dir.name,
                label="normal",
                start=normal_start,
                end=normal_end,
                metadata_path=metadata_path.resolve(),
            )
        )

    if not runs:
        raise FileNotFoundError(f"No normal runs found for {telemetry_day}")
    return runs


def _iter_hour_starts(windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[pd.Timestamp]:
    hours = set()
    for start, end in windows:
        current = start.floor("h")
        final = end.floor("h")
        while current <= final:
            hours.add(current)
            current += pd.Timedelta(hours=1)
    return sorted(hours)


def select_hourly_trace_files(traces_dir: Path, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[Path]:
    selected = []
    for hour_start in _iter_hour_starts(windows):
        file_path = traces_dir / f"jaeger_traces_parsed_{hour_start.strftime('%H')}.csv"
        if file_path.exists():
            selected.append(file_path)
    if not selected:
        raise FileNotFoundError(f"No hourly trace files found in {traces_dir}")
    return selected


def build_trace_mask(series: pd.Series, windows_us: list[tuple[int, int]]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for start_us, end_us in windows_us:
        mask |= (series >= start_us) & (series <= end_us)
    return mask


def load_trace_window(cache: HourlyTraceCache, traces_dir: Path, run: RunWindow) -> pd.DataFrame:
    windows = [(run.start, run.end)]
    windows_us = [(timestamp_to_us(start), timestamp_to_us(end)) for start, end in windows]
    selected_files = select_hourly_trace_files(traces_dir, windows)
    selected_frames: list[pd.DataFrame] = []
    for trace_file in selected_files:
        frame = cache.load_file(trace_file)
        if frame.empty:
            continue
        windowed = frame[build_trace_mask(frame["timestamp"], windows_us)]
        if not windowed.empty:
            selected_frames.append(windowed)
    if not selected_frames:
        return pd.DataFrame(columns=TRACE_USECOLS)
    return pd.concat(selected_frames, ignore_index=True)


def build_trace_groups(df: pd.DataFrame) -> dict[str, list[dict]]:
    if df.empty:
        return {}
    groups: dict[str, list[dict]] = {}
    for trace_id, group in df.groupby("trace_id", sort=False):
        groups[str(trace_id)] = group.sort_values("timestamp").to_dict("records")
    return groups


def split_trace_groups(
    trace_groups: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    keys = list(trace_groups.keys())
    if len(keys) <= 1:
        return trace_groups, dict(trace_groups)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(keys))
    rng.shuffle(indices)
    valid_size = max(1, int(len(indices) * VALIDATION_SPLIT))
    if valid_size >= len(indices):
        valid_size = len(indices) - 1

    valid_keys = {keys[index] for index in indices[:valid_size]}
    train_groups = {key: value for key, value in trace_groups.items() if key not in valid_keys}
    valid_groups = {key: value for key, value in trace_groups.items() if key in valid_keys}
    if not valid_groups:
        valid_groups = dict(train_groups)
    return train_groups, valid_groups


def _parent_missing(value: object) -> bool:
    return pd.isna(value) or value == ""


def extract_call_paths(spans: list[dict]) -> list[tuple[str, tuple[str, ...], float]]:
    span_map = {str(span["span_id"]): span for span in spans}
    server_spans = [span for span in spans if str(span.get("span_kind", "")).lower() == "server"]
    target_spans = server_spans if server_spans else spans
    result: list[tuple[str, tuple[str, ...], float]] = []

    for span in sorted(target_spans, key=lambda item: item["timestamp"]):
        services_reversed: list[str] = []
        current = span
        last_service = None
        while current is not None:
            service = str(current["service"])
            if service != last_service:
                services_reversed.append(service)
                last_service = service
            parent_id = current.get("parent_span_id")
            if _parent_missing(parent_id):
                break
            current = span_map.get(str(parent_id))
        path = tuple(reversed(services_reversed))
        result.append((str(span["service"]), path, float(span["duration"])))
    return result


def build_call_path_vocabulary(trace_groups: dict[str, list[dict]]) -> list[dict]:
    seen = set()
    for spans in trace_groups.values():
        for service, path, _ in extract_call_paths(spans):
            seen.add((service, path))
    return [{"service": service, "call_path": list(path)} for service, path in sorted(seen)]


def build_path_index(call_path_list: list[dict]) -> dict[tuple[str, tuple[str, ...]], int]:
    return {
        (str(item["service"]), tuple(item["call_path"])): index
        for index, item in enumerate(call_path_list)
    }


def build_stv_vector(
    spans: list[dict],
    path_index: dict[tuple[str, tuple[str, ...]], int],
    dim: int,
) -> tuple[np.ndarray, int]:
    vector = np.full(dim, -1.0, dtype=np.float32)
    unseen_count = 0
    for service, path, duration in extract_call_paths(spans):
        idx = path_index.get((service, path))
        if idx is None:
            unseen_count += 1
            continue
        vector[idx] = max(vector[idx], duration) if vector[idx] >= 0 else duration
    return vector, unseen_count


def build_stv_matrix(
    trace_groups: dict[str, list[dict]],
    call_path_list: list[dict],
) -> tuple[np.ndarray, list[str], list[int]]:
    path_index = build_path_index(call_path_list)
    dim = len(call_path_list)
    vectors: list[np.ndarray] = []
    trace_ids: list[str] = []
    unseen_counts: list[int] = []

    for trace_id, spans in trace_groups.items():
        vector, unseen_count = build_stv_vector(spans, path_index, dim)
        if int((vector != -1).sum()) < MIN_VALID_TRACE_DIMS:
            continue
        vectors.append(vector)
        trace_ids.append(trace_id)
        unseen_counts.append(unseen_count)

    if not vectors:
        return np.empty((0, dim), dtype=np.float32), [], []
    return np.vstack(vectors), trace_ids, unseen_counts


def compute_normalization_stats(stv_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = stv_matrix.shape[1]
    means = np.zeros(dim, dtype=np.float32)
    stds = np.ones(dim, dtype=np.float32)
    for i in range(dim):
        valid = stv_matrix[stv_matrix[:, i] > 1e-5, i]
        if valid.size == 0:
            continue
        means[i] = float(valid.mean())
        stds[i] = max(1.0, float(valid.std()))
    return means, stds


def normalize_stv(stv_matrix: np.ndarray, means: np.ndarray, stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (stv_matrix > 1e-5).astype(np.float32)
    normalized = np.full_like(stv_matrix, -1.0, dtype=np.float32)
    for i in range(stv_matrix.shape[1]):
        valid = mask[:, i] == 1
        normalized[valid, i] = (stv_matrix[valid, i] - means[i]) / stds[i]
    return normalized, mask


def mask_key_from_vector(stv: np.ndarray) -> bytes:
    mask = (stv != -1).astype(np.uint8)
    return np.packbits(mask).tobytes()


def build_hstv_index(stv_matrix: np.ndarray) -> dict[bytes, np.ndarray]:
    grouped: dict[bytes, list[int]] = {}
    for index, stv in enumerate(stv_matrix):
        grouped.setdefault(mask_key_from_vector(stv), []).append(index)
    return {key: np.asarray(indices, dtype=int) for key, indices in grouped.items()}


def fit_kde_tail_model(values: np.ndarray) -> KdeTailModel:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    if flattened.size == 0:
        raise ValueError("Cannot fit KDE on empty values.")
    if flattened.size == 1:
        center = flattened[0]
        grid = np.linspace(center - 1.0, center + 1.0, KDE_GRID_SIZE)
        cdf = np.linspace(0.0, 1.0, KDE_GRID_SIZE)
        return KdeTailModel(values=flattened, grid=grid, cdf=cdf, bandwidth=1.0)

    if np.allclose(flattened, flattened[0]):
        center = flattened[0]
        grid = np.linspace(center - 1.0, center + 1.0, KDE_GRID_SIZE)
        cdf = np.linspace(0.0, 1.0, KDE_GRID_SIZE)
        return KdeTailModel(values=flattened, grid=grid, cdf=cdf, bandwidth=1.0)

    kde = gaussian_kde(flattened)
    bandwidth = float(np.sqrt(kde.covariance.squeeze()))
    margin = max(3.0 * bandwidth, 1.0)
    grid = np.linspace(flattened.min() - margin, flattened.max() + margin, KDE_GRID_SIZE)
    density = kde(grid)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]
    return KdeTailModel(values=flattened, grid=grid, cdf=cdf, bandwidth=bandwidth)


def kde_left_tail_pvalue(model: KdeTailModel, value: float) -> float:
    return float(np.interp(value, model.grid, model.cdf, left=0.0, right=1.0))


def train_traceanomaly_model(
    normalized_train: np.ndarray,
    train_mask: np.ndarray,
    normalized_valid: np.ndarray,
    valid_mask: np.ndarray,
    input_dim: int,
) -> tuple[PosteriorFlowVAE, list[float]]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(
        torch.tensor(normalized_train, dtype=torch.float32),
        torch.tensor(train_mask, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = PosteriorFlowVAE(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        flow_layers=FLOW_LAYERS,
        flow_hidden_dim=FLOW_HIDDEN_DIM,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_ANNEAL_EPOCH_FREQ, gamma=LR_ANNEAL_FACTOR)
    training_losses: list[float] = []
    best_valid_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_eval_count = 0

    valid_x_tensor = torch.tensor(normalized_valid, dtype=torch.float32).to(device)
    valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_mask in dataloader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            elbo, _ = model.elbo(batch_x, batch_mask)
            loss = -elbo.mean()
            if not torch.isfinite(loss):
                raise ValueError(f"TraceAnomaly training diverged: non-finite loss at epoch {epoch + 1}")
            loss.backward()
            if GRAD_NORM_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
        training_losses.append(total_loss / max(len(dataloader), 1))
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                valid_log_likelihood = model.estimate_log_likelihood(valid_x_tensor, valid_mask_tensor, IMPORTANCE_SAMPLES)
                valid_loss = float((-valid_log_likelihood.mean()).item())
            if not math.isfinite(valid_loss):
                raise ValueError(f"TraceAnomaly validation diverged: non-finite loss at epoch {epoch + 1}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_eval_count = 0
            else:
                bad_eval_count += 1

            print(
                f"[INFO] epoch={epoch + 1}/{EPOCHS} "
                f"loss={training_losses[-1]:.6f} valid_loss={valid_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.6g}"
            )
            if bad_eval_count >= EARLY_STOPPING_PATIENCE_EVALS:
                print(f"[INFO] Early stopping at epoch {epoch + 1} after {bad_eval_count} non-improving validations")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.to(torch.device("cpu")).eval(), training_losses


def estimate_log_likelihoods(
    model: PosteriorFlowVAE,
    normalized_stv: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    if normalized_stv.size == 0:
        return np.empty((0,), dtype=float)

    x_tensor = torch.tensor(normalized_stv, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            batch_x = x_tensor[start : start + batch_size]
            batch_mask = mask_tensor[start : start + batch_size]
            log_likelihood = model.estimate_log_likelihood(batch_x, batch_mask, IMPORTANCE_SAMPLES)
            outputs.append(log_likelihood.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def split_train_valid_stv(stv_matrix: np.ndarray, trace_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if len(stv_matrix) <= 1:
        return stv_matrix, stv_matrix.copy(), trace_ids, trace_ids.copy()

    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(stv_matrix))
    rng.shuffle(indices)
    valid_size = max(1, int(len(indices) * VALIDATION_SPLIT))
    if valid_size >= len(indices):
        valid_size = len(indices) - 1

    valid_indices = np.sort(indices[:valid_size])
    train_indices = np.sort(indices[valid_size:])
    if len(train_indices) == 0:
        train_indices = valid_indices[:1]
        valid_indices = valid_indices[1:]

    train_ids = [trace_ids[index] for index in train_indices]
    valid_ids = [trace_ids[index] for index in valid_indices] if len(valid_indices) > 0 else train_ids.copy()
    valid_matrix = stv_matrix[valid_indices] if len(valid_indices) > 0 else stv_matrix[train_indices].copy()
    return stv_matrix[train_indices], valid_matrix, train_ids, valid_ids


def empirical_left_tail_pvalues(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_scores, dtype=float)
    query = np.asarray(scores, dtype=float)
    if reference.size == 0:
        return np.full(query.shape, np.nan, dtype=float)
    return np.asarray([(reference <= value).mean() for value in query], dtype=float)


def score_traces(
    model: PosteriorFlowVAE,
    stv_matrix: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    trace_ids: list[str],
    unseen_counts: list[int],
    validation_scores: np.ndarray,
    validation_score_threshold: float,
) -> pd.DataFrame:
    if stv_matrix.size == 0:
        return pd.DataFrame(
            columns=[
                "trace_id",
                "score",
                "p_value",
                "known_dim_count",
                "unseen_path_count",
                "is_anomalous",
                "anomaly_reason",
                "trace_rank",
            ]
        )

    normalized, mask = normalize_stv(stv_matrix, means, stds)
    log_likelihoods = estimate_log_likelihoods(model, normalized, mask) / max(stv_matrix.shape[1], 1)
    known_dim_count = mask.sum(axis=1).astype(int)
    p_values = empirical_left_tail_pvalues(validation_scores, log_likelihoods)
    unseen_array = np.asarray(unseen_counts, dtype=int)
    is_anomalous = (unseen_array > 0) | (log_likelihoods < validation_score_threshold)
    anomaly_reason = np.where(
        unseen_array > 0,
        "unseen_call_path",
        np.where(log_likelihoods < validation_score_threshold, "score_below_validation_min", "normal"),
    )

    result = (
        pd.DataFrame(
            {
                "trace_id": trace_ids,
                "score": log_likelihoods.astype(float),
                "p_value": p_values,
                "known_dim_count": known_dim_count,
                "unseen_path_count": unseen_array,
                "is_anomalous": is_anomalous,
                "anomaly_reason": anomaly_reason,
            }
        )
        .sort_values(["is_anomalous", "score", "p_value"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    result["trace_rank"] = np.arange(1, len(result) + 1)
    result["validation_score_threshold"] = float(validation_score_threshold)
    return result


def unseen_root_cause_candidate(
    spans: list[dict],
    path_index: dict[tuple[str, tuple[str, ...]], int],
) -> dict | None:
    unseen = []
    for service, path, duration in extract_call_paths(spans):
        if (service, path) not in path_index:
            unseen.append(
                {
                    "service": service,
                    "call_path": list(path),
                    "path_length": len(path),
                    "score": float(duration),
                    "method": "unseen_call_path",
                }
            )
    if not unseen:
        return None
    unseen.sort(key=lambda item: (-item["path_length"], -item["score"], item["service"]))
    return unseen[0]


def _common_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    length = 0
    for left_service, right_service in zip(left, right):
        if left_service != right_service:
            break
        length += 1
    return length


def longest_common_path_root_cause(spans: list[dict], call_path_list: list[dict]) -> dict | None:
    known_paths = [tuple(item["call_path"]) for item in call_path_list if item.get("call_path")]
    if not known_paths:
        return None

    candidate_paths = sorted(
        extract_call_paths(spans),
        key=lambda item: (-len(item[1]), -item[2], item[0]),
    )
    if not candidate_paths:
        return None

    for _, path, duration in candidate_paths:
        best_prefix_len = max((_common_prefix_length(path, known_path) for known_path in known_paths), default=0)
        if best_prefix_len <= 0 or best_prefix_len >= len(path):
            continue

        inferred_path = path[: best_prefix_len + 1]
        return {
            "service": inferred_path[-1],
            "call_path": list(inferred_path),
            "path_length": len(inferred_path),
            "score": float(duration),
            "method": "paper_longest_common_path_fallback",
        }
    return None


def _build_trace_candidate_diagnostics(
    chosen_candidate: dict,
    all_candidates: list[dict],
) -> tuple[list[dict], list[dict], list[str]]:
    normalized_candidates = []
    seen_paths: set[tuple[str, tuple[str, ...], str]] = set()
    for index, candidate in enumerate(all_candidates):
        service = str(candidate.get("service") or "")
        call_path = tuple(str(item) for item in candidate.get("call_path", []))
        method = str(candidate.get("method") or "")
        dedupe_key = (service, call_path, method)
        if dedupe_key in seen_paths:
            continue
        seen_paths.add(dedupe_key)
        candidate_copy = {
            "service": service,
            "call_path": list(call_path),
            "path_length": int(candidate.get("path_length", len(call_path))),
            "score": float(candidate.get("score", 0.0)),
            "method": method,
            "is_paper_style_chosen_root_cause": index == 0,
        }
        normalized_candidates.append(candidate_copy)

    ordered_candidates = []
    ordered_service_candidates = []
    seen_services: set[str] = set()
    for candidate in normalized_candidates:
        service = str(candidate.get("service") or "")
        if not service or service in seen_services:
            continue
        if len(ordered_candidates) >= 1 + MAX_SECONDARY_DIMENSIONS_PER_TRACE:
            break
        ordered_candidates.append(candidate)
        ordered_service_candidates.append(service)
        seen_services.add(service)
    return normalized_candidates, ordered_candidates, ordered_service_candidates


def _build_localization_result(
    trace_id: str,
    chosen_candidate: dict,
    score_row: pd.Series,
    method: str,
    reference_trace_count: int,
    all_candidates: list[dict] | None = None,
) -> dict:
    candidate_pool = all_candidates if all_candidates is not None else [chosen_candidate]
    normalized_candidates, ordered_candidates, ordered_service_candidates = _build_trace_candidate_diagnostics(
        chosen_candidate=chosen_candidate,
        all_candidates=candidate_pool,
    )
    chosen_root_cause = ordered_candidates[0] if ordered_candidates else {
        "service": str(chosen_candidate.get("service") or ""),
        "call_path": list(chosen_candidate.get("call_path", [])),
        "path_length": int(chosen_candidate.get("path_length", 0)),
        "score": float(chosen_candidate.get("score", 0.0)),
        "method": str(chosen_candidate.get("method") or method),
        "is_paper_style_chosen_root_cause": True,
    }
    return {
        "trace_id": trace_id,
        "service": chosen_root_cause["service"],
        "call_path": chosen_root_cause["call_path"],
        "path_length": chosen_root_cause["path_length"],
        "score": chosen_root_cause["score"],
        "p_value": float(score_row["p_value"]),
        "trace_score": float(score_row["score"]),
        "method": method,
        "reference_trace_count": reference_trace_count,
        "chosen_root_cause": chosen_root_cause,
        "all_anomalous_dimensions": normalized_candidates,
        "ordered_candidates": ordered_candidates,
        "ordered_service_candidates": ordered_service_candidates,
    }


def _candidate_vote_entries(candidate: dict) -> list[tuple[str, float]]:
    path = [str(item) for item in candidate.get("call_path", []) if str(item)]
    if not path:
        return []

    score = float(candidate.get("score", 0.0))
    leaf_service = str(path[-1])

    if SERVICE_RANKING_MODE == "leaf_only":
        return [(leaf_service, score)]

    if SERVICE_RANKING_MODE == "leaf_decay":
        return [(str(service), score / depth) for depth, service in enumerate(reversed(path), start=1)]

    if SERVICE_RANKING_MODE == "path_vote_hybrid":
        path_length = len(path)
        weighted = []
        for depth, service in enumerate(path, start=1):
            weight = score * (SERVICE_PATH_ANCESTOR_WEIGHT / depth)
            if depth == path_length:
                weight += score * (1.0 - SERVICE_PATH_ANCESTOR_WEIGHT)
            weighted.append((str(service), weight))
        return weighted

    raise ValueError(f"Unsupported SERVICE_RANKING_MODE: {SERVICE_RANKING_MODE}")


def localize_trace_root_cause(
    trace_id: str,
    spans: list[dict],
    stv: np.ndarray,
    score_row: pd.Series,
    normal_stv_matrix: np.ndarray,
    hstv_index: dict[bytes, np.ndarray],
    call_path_list: list[dict],
    path_index: dict[tuple[str, tuple[str, ...]], int],
    global_means: np.ndarray,
    global_stds: np.ndarray,
) -> dict | None:
    if int(score_row["unseen_path_count"]) > 0:
        unseen_candidate = unseen_root_cause_candidate(spans, path_index)
        if unseen_candidate is not None:
            return _build_localization_result(
                trace_id=trace_id,
                chosen_candidate=unseen_candidate,
                score_row=score_row,
                method=str(unseen_candidate["method"]),
                reference_trace_count=0,
            )

    reference_indices = hstv_index.get(mask_key_from_vector(stv))
    method = "paper_hstv_3sigma"
    reference_matrix = None
    if reference_indices is not None and len(reference_indices) >= MIN_HSTV_REFERENCE_TRACES:
        reference_matrix = normal_stv_matrix[reference_indices]
    else:
        longest_common_candidate = longest_common_path_root_cause(spans, call_path_list)
        if longest_common_candidate is not None:
            return _build_localization_result(
                trace_id=trace_id,
                chosen_candidate=longest_common_candidate,
                score_row=score_row,
                method=str(longest_common_candidate["method"]),
                reference_trace_count=0,
            )
        method = "global_3sigma_fallback"
        reference_indices = None

    anomalous_dimensions = []
    valid_dims = np.where(stv != -1)[0]
    for dim in valid_dims:
        if reference_matrix is not None:
            valid_ref = reference_matrix[reference_matrix[:, dim] != -1, dim]
            if valid_ref.size == 0:
                continue
            mean = float(valid_ref.mean())
            std = float(valid_ref.std())
            std = std if std > 1e-8 else 1.0
        else:
            mean = float(global_means[dim])
            std = float(global_stds[dim]) if global_stds[dim] > 1e-8 else 1.0

        value = float(stv[dim])
        if value > mean + ANOMALOUS_STD_THRESHOLD * std or value < mean - ANOMALOUS_STD_THRESHOLD * std:
            anomalous_dimensions.append(
                {
                    "dim": dim,
                    "service": str(call_path_list[dim]["service"]),
                    "call_path": call_path_list[dim]["call_path"],
                    "path_length": len(call_path_list[dim]["call_path"]),
                    "score": abs((value - mean) / std),
                }
            )

    if not anomalous_dimensions:
        return None

    anomalous_dimensions.sort(key=lambda item: (-item["path_length"], -item["score"], item["service"]))
    chosen = anomalous_dimensions[0]
    candidate_pool = anomalous_dimensions if USE_SECONDARY_ANOMALOUS_DIMENSIONS else [chosen]
    return _build_localization_result(
        trace_id=trace_id,
        chosen_candidate=chosen,
        score_row=score_row,
        method=method,
        reference_trace_count=int(len(reference_indices)) if reference_indices is not None else int(len(normal_stv_matrix)),
        all_candidates=candidate_pool,
    )


def localize_fault_run(
    trace_groups: dict[str, list[dict]],
    stv_matrix: np.ndarray,
    trace_ids: list[str],
    trace_scores_df: pd.DataFrame,
    normal_stv_matrix: np.ndarray,
    hstv_index: dict[bytes, np.ndarray],
    call_path_list: list[dict],
    global_means: np.ndarray,
    global_stds: np.ndarray,
) -> pd.DataFrame:
    if trace_scores_df.empty or stv_matrix.size == 0:
        return pd.DataFrame(
            columns=[
                "trace_id",
                "service",
                "call_path",
                "path_length",
                "score",
                "p_value",
                "trace_score",
                "method",
                "reference_trace_count",
                "chosen_root_cause",
                "all_anomalous_dimensions",
                "ordered_candidates",
                "ordered_service_candidates",
            ]
        )

    path_index = build_path_index(call_path_list)
    trace_index = {trace_id: index for index, trace_id in enumerate(trace_ids)}
    rows = []

    anomalous_rows = trace_scores_df[trace_scores_df["is_anomalous"]].copy()
    if anomalous_rows.empty and not trace_scores_df.empty:
        anomalous_rows = trace_scores_df.head(1).copy()

    for score_row in anomalous_rows.itertuples(index=False):
        trace_id = str(score_row.trace_id)
        idx = trace_index.get(trace_id)
        if idx is None:
            continue
        row_dict = localize_trace_root_cause(
            trace_id=trace_id,
            spans=trace_groups.get(trace_id, []),
            stv=stv_matrix[idx],
            score_row=pd.Series(score_row._asdict()),
            normal_stv_matrix=normal_stv_matrix,
            hstv_index=hstv_index,
            call_path_list=call_path_list,
            path_index=path_index,
            global_means=global_means,
            global_stds=global_stds,
        )
        if row_dict is not None:
            rows.append(row_dict)

    if not rows:
        return pd.DataFrame(
            columns=[
                "trace_id",
                "service",
                "call_path",
                "path_length",
                "score",
                "p_value",
                "trace_score",
                "method",
                "reference_trace_count",
                "chosen_root_cause",
                "all_anomalous_dimensions",
                "ordered_candidates",
                "ordered_service_candidates",
            ]
        )
    return pd.DataFrame(rows).sort_values(["path_length", "score"], ascending=[False, False]).reset_index(drop=True)


def aggregate_service_ranking(localization_df: pd.DataFrame) -> pd.DataFrame:
    if localization_df.empty:
        return pd.DataFrame(columns=["service", "score", "trace_vote_count"])

    votes: dict[str, float] = {}
    trace_vote_counts: dict[str, int] = {}
    chosen_trace_vote_counts: dict[str, int] = {}
    secondary_trace_vote_counts: dict[str, int] = {}
    chosen_scores: dict[str, float] = {}
    secondary_scores: dict[str, float] = {}

    for row in localization_df.itertuples(index=False):
        if USE_SECONDARY_ANOMALOUS_DIMENSIONS and isinstance(getattr(row, "ordered_candidates", None), list):
            candidate_list = row.ordered_candidates
        else:
            candidate_list = [
                {
                    "service": str(row.service),
                    "call_path": row.call_path if isinstance(row.call_path, list) else [],
                    "path_length": int(row.path_length),
                    "score": float(row.score),
                    "method": str(row.method),
                    "is_paper_style_chosen_root_cause": True,
                }
            ]

        for candidate_index, candidate in enumerate(candidate_list):
            candidate_weight = 1.0 if candidate_index == 0 else SECONDARY_DIMENSION_WEIGHT
            if candidate_index > 0 and not USE_SECONDARY_ANOMALOUS_DIMENSIONS:
                continue
            for service, weighted_score in _candidate_vote_entries(candidate):
                contribution = float(weighted_score) * candidate_weight
                if contribution <= 0:
                    continue
                votes[service] = votes.get(service, 0.0) + contribution
                trace_vote_counts[service] = trace_vote_counts.get(service, 0) + 1
                if candidate_index == 0:
                    chosen_trace_vote_counts[service] = chosen_trace_vote_counts.get(service, 0) + 1
                    chosen_scores[service] = chosen_scores.get(service, 0.0) + contribution
                else:
                    secondary_trace_vote_counts[service] = secondary_trace_vote_counts.get(service, 0) + 1
                    secondary_scores[service] = secondary_scores.get(service, 0.0) + contribution

    rows = [
        {
            "service": service,
            "score": votes[service],
            "trace_vote_count": trace_vote_counts.get(service, 0),
            "chosen_trace_vote_count": chosen_trace_vote_counts.get(service, 0),
            "secondary_trace_vote_count": secondary_trace_vote_counts.get(service, 0),
            "chosen_score": chosen_scores.get(service, 0.0),
            "secondary_score": secondary_scores.get(service, 0.0),
        }
        for service in votes
    ]
    ranking_df = pd.DataFrame(rows)
    if SERVICE_RANKING_SORT_MODE == "count_then_score":
        ranking_df = ranking_df.sort_values(
            ["chosen_trace_vote_count", "chosen_score", "score", "secondary_trace_vote_count", "service"],
            ascending=[False, False, False, False, True],
        )
    elif SERVICE_RANKING_SORT_MODE == "score_only":
        ranking_df = ranking_df.sort_values(
            ["chosen_score", "score", "chosen_trace_vote_count", "secondary_trace_vote_count", "service"],
            ascending=[False, False, False, False, True],
        )
    else:
        raise ValueError(f"Unsupported SERVICE_RANKING_SORT_MODE: {SERVICE_RANKING_SORT_MODE}")
    return ranking_df.reset_index(drop=True)[["service", "score", "trace_vote_count"]]


def build_service_frequency_prior(call_path_list: list[dict]) -> list[str]:
    counts: Counter[str] = Counter()
    for item in call_path_list:
        for service in item.get("call_path", []):
            counts[str(service)] += 1
    return [
        service
        for service, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def annotate_path_only_topk_fill(localization_df: pd.DataFrame) -> pd.DataFrame:
    if localization_df.empty:
        return localization_df

    rows = []
    for row in localization_df.to_dict("records"):
        original_candidates = [
            str(service)
            for service in row.get("ordered_service_candidates", [])
            if str(service)
        ]
        filled_candidates = list(original_candidates)
        added_services: list[str] = []

        if TOPK_FILL_ENABLED and TOPK_FILL_MODE == "path_ancestors_only" and len(filled_candidates) < TOPK_FILL_K:
            for candidate in row.get("ordered_candidates", []) or []:
                path = [str(service) for service in candidate.get("call_path", []) if str(service)]
                if len(path) <= 1:
                    continue
                for ancestor_service in reversed(path[:-1]):
                    if ancestor_service in filled_candidates:
                        continue
                    filled_candidates.append(ancestor_service)
                    added_services.append(ancestor_service)
                    if len(filled_candidates) >= TOPK_FILL_K:
                        break
                if len(filled_candidates) >= TOPK_FILL_K:
                    break

        row["original_ordered_service_candidates"] = original_candidates
        row["filled_service_candidates"] = filled_candidates
        row["ancestor_fill_triggered"] = bool(added_services)
        row["ancestor_fill_added_services"] = added_services
        rows.append(row)

    return pd.DataFrame(rows)


def build_output_service_candidates(
    service_ranking_df: pd.DataFrame,
    localization_df: pd.DataFrame,
) -> tuple[list[str], bool, list[str]]:
    ranked = [str(service) for service in service_ranking_df["service"].tolist() if str(service)]
    if not TOPK_FILL_ENABLED or TOPK_FILL_MODE != "path_ancestors_only" or len(ranked) >= TOPK_FILL_K:
        return ranked, False, []

    filled_ranked = list(ranked)
    seen_services = set(filled_ranked)
    added_services: list[str] = []
    for row in localization_df.itertuples(index=False):
        candidate_services = getattr(row, "filled_service_candidates", None)
        if not isinstance(candidate_services, list):
            continue
        for service in candidate_services:
            service_name = str(service)
            if not service_name or service_name in seen_services:
                continue
            filled_ranked.append(service_name)
            seen_services.add(service_name)
            added_services.append(service_name)
            if len(filled_ranked) >= TOPK_FILL_K:
                return filled_ranked, True, added_services
    return filled_ranked, bool(added_services), added_services


def evaluate_topk(
    service_ranking_df: pd.DataFrame,
    ground_truth_service: str,
    localization_df: pd.DataFrame,
) -> dict:
    ranked, fill_triggered, added_services = build_output_service_candidates(service_ranking_df, localization_df)
    return {
        "predicted_service_top1": ranked[0] if ranked else None,
        "predicted_service_top3": ranked[:3],
        "predicted_service_top5": ranked[:5],
        "service_top1_hit": ground_truth_service in ranked[:1],
        "service_top3_hit": ground_truth_service in ranked[:3],
        "service_top5_hit": ground_truth_service in ranked[:5],
        "output_service_candidates_before_fill": service_ranking_df["service"].tolist(),
        "output_service_candidates_after_fill": ranked,
        "topk_fill_triggered": fill_triggered,
        "topk_fill_added_services": added_services,
    }


def save_normal_baseline(
    output_dir: Path,
    call_path_list: list[dict],
    means: np.ndarray,
    stds: np.ndarray,
    training_losses: list[float],
    model: PosteriorFlowVAE,
    train_trace_count: int,
    validation_trace_count: int,
    normal_run_count: int,
    training_log_likelihoods: np.ndarray,
    validation_scores: np.ndarray,
    validation_score_threshold: float,
    telemetry_day: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CALL_PATH_FILE).write_text(json.dumps(call_path_list, indent=2), encoding="utf-8")
    (output_dir / NORMALIZATION_FILE).write_text(
        json.dumps({"mean": means.tolist(), "std": stds.tolist()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / TRAINING_LOSS_FILE).write_text(json.dumps(training_losses, indent=2), encoding="utf-8")
    (output_dir / NORMAL_BASELINE_SUMMARY_FILE).write_text(
        json.dumps(
            {
                "telemetry_day": telemetry_day,
                "normal_run_count": normal_run_count,
                "normal_trace_count": train_trace_count + validation_trace_count,
                "train_trace_count": train_trace_count,
                "validation_trace_count": validation_trace_count,
                "call_path_count": len(call_path_list),
                "p_value_alpha": P_VALUE_ALPHA,
                "anomalous_std_threshold": ANOMALOUS_STD_THRESHOLD,
                "importance_samples": IMPORTANCE_SAMPLES,
                "flow_layers": FLOW_LAYERS,
                "service_ranking_mode": SERVICE_RANKING_MODE,
                "service_ranking_sort_mode": SERVICE_RANKING_SORT_MODE,
                "service_path_ancestor_weight": SERVICE_PATH_ANCESTOR_WEIGHT,
                "use_secondary_anomalous_dimensions": USE_SECONDARY_ANOMALOUS_DIMENSIONS,
                "secondary_dimension_weight": SECONDARY_DIMENSION_WEIGHT,
                "max_secondary_dimensions_per_trace": MAX_SECONDARY_DIMENSIONS_PER_TRACE,
                "topk_fill_enabled": TOPK_FILL_ENABLED,
                "topk_fill_k": TOPK_FILL_K,
                "topk_fill_mode": TOPK_FILL_MODE,
                "score_method": "paper_log_likelihood_per_dimension",
                "validation_score_threshold": float(validation_score_threshold),
                "train_log_likelihood_mean": float(training_log_likelihoods.mean()),
                "train_log_likelihood_std": float(training_log_likelihoods.std()),
                "validation_log_likelihood_mean": float(validation_scores.mean()) if len(validation_scores) else np.nan,
                "validation_log_likelihood_std": float(validation_scores.std()) if len(validation_scores) else np.nan,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model_dir = output_dir / MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "traceanomaly_model.pth")


def save_run_outputs(
    output_dir: Path,
    trace_scores_df: pd.DataFrame,
    localization_df: pd.DataFrame,
    service_ranking_df: pd.DataFrame,
    run_summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_scores_df.to_csv(output_dir / TRACE_SCORES_FILE, index=False)
    localization_save_df = localization_df.copy()
    for column in [
        "call_path",
        "chosen_root_cause",
        "all_anomalous_dimensions",
        "ordered_candidates",
        "ordered_service_candidates",
        "original_ordered_service_candidates",
        "filled_service_candidates",
        "ancestor_fill_added_services",
    ]:
        if column in localization_save_df.columns:
            localization_save_df[column] = localization_save_df[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            )
    localization_save_df.to_csv(output_dir / TRACE_LOCALIZATION_FILE, index=False)
    service_ranking_df.to_csv(output_dir / SERVICE_RANKING_FILE, index=False)
    (output_dir / RUN_SUMMARY_FILE).write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_day_outputs(
    output_dir: Path,
    detail_rows: list[dict],
    total_runtime_seconds: float,
    unclear_points: list[str],
    telemetry_day: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_df = pd.DataFrame(detail_rows).sort_values("exp_id").reset_index(drop=True)
    details_df.to_csv(output_dir / DAY_DETAILS_FILE, index=False)

    n_total = len(details_df)
    summary_df = pd.DataFrame(
        [
            {
                "telemetry_day": telemetry_day,
                "n_total": n_total,
                "n_ok": int(details_df["predicted_service_top1"].notna().sum()) if n_total else 0,
                "n_error": int(details_df["predicted_service_top1"].isna().sum()) if n_total else 0,
                "service_top1_accuracy": float(details_df["service_top1_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top3_accuracy": float(details_df["service_top3_hit"].fillna(False).mean()) if n_total else np.nan,
                "service_top5_accuracy": float(details_df["service_top5_hit"].fillna(False).mean()) if n_total else np.nan,
                "total_runtime_seconds": total_runtime_seconds,
                "avg_runtime_per_exception_seconds": total_runtime_seconds / n_total if n_total else np.nan,
                "sum_of_individual_runtime_seconds": float(details_df["runtime_seconds"].fillna(0).sum()) if n_total else np.nan,
                "avg_runtime_per_processed_exception_seconds": float(details_df["runtime_seconds"].mean()) if n_total else np.nan,
                "service_ranking_mode": SERVICE_RANKING_MODE,
                "service_ranking_sort_mode": SERVICE_RANKING_SORT_MODE,
                "service_path_ancestor_weight": SERVICE_PATH_ANCESTOR_WEIGHT,
                "use_secondary_anomalous_dimensions": USE_SECONDARY_ANOMALOUS_DIMENSIONS,
                "secondary_dimension_weight": SECONDARY_DIMENSION_WEIGHT,
                "max_secondary_dimensions_per_trace": MAX_SECONDARY_DIMENSIONS_PER_TRACE,
                "topk_fill_enabled": TOPK_FILL_ENABLED,
                "topk_fill_k": TOPK_FILL_K,
                "topk_fill_mode": TOPK_FILL_MODE,
            }
        ]
    )
    summary_df.to_csv(output_dir / DAY_SUMMARY_FILE, index=False)
    (output_dir / UNCLEAR_POINTS_FILE).write_text(json.dumps({"unclear_points": unclear_points}, indent=2), encoding="utf-8")
    return details_df, summary_df


def build_all_days_outputs(
    output_root: Path,
    all_details_dfs: list[pd.DataFrame],
    all_summary_dfs: list[pd.DataFrame],
    total_script_runtime_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_details_df = pd.concat(all_details_dfs, ignore_index=True) if all_details_dfs else pd.DataFrame()
    combined_summary_df = pd.concat(all_summary_dfs, ignore_index=True) if all_summary_dfs else pd.DataFrame()

    if combined_details_df.empty:
        overall_summary_df = pd.DataFrame(
            columns=[
                "telemetry_day",
                "n_total",
                "n_ok",
                "n_error",
                "service_top1_accuracy",
                "service_top3_accuracy",
                "service_top5_accuracy",
                "total_runtime_seconds",
                "avg_runtime_per_exception_seconds",
                "sum_of_individual_runtime_seconds",
                "avg_runtime_per_processed_exception_seconds",
                "service_ranking_mode",
                "service_ranking_sort_mode",
                "service_path_ancestor_weight",
            ]
        )
    else:
        n_total = len(combined_details_df)
        overall_summary_df = pd.DataFrame(
            [
                {
                    "telemetry_day": "ALL_DAYS",
                    "n_total": n_total,
                    "n_ok": int(combined_details_df["predicted_service_top1"].notna().sum()),
                    "n_error": int(combined_details_df["predicted_service_top1"].isna().sum()),
                    "service_top1_accuracy": float(combined_details_df["service_top1_hit"].fillna(False).mean()),
                    "service_top3_accuracy": float(combined_details_df["service_top3_hit"].fillna(False).mean()),
                    "service_top5_accuracy": float(combined_details_df["service_top5_hit"].fillna(False).mean()),
                    "total_runtime_seconds": total_script_runtime_seconds,
                    "avg_runtime_per_exception_seconds": total_script_runtime_seconds / n_total if n_total else np.nan,
                    "sum_of_individual_runtime_seconds": float(combined_details_df["runtime_seconds"].fillna(0).sum()),
                    "avg_runtime_per_processed_exception_seconds": float(combined_details_df["runtime_seconds"].mean()),
                    "service_ranking_mode": SERVICE_RANKING_MODE,
                    "service_ranking_sort_mode": SERVICE_RANKING_SORT_MODE,
                    "service_path_ancestor_weight": SERVICE_PATH_ANCESTOR_WEIGHT,
                }
            ]
        )

    output_root.mkdir(parents=True, exist_ok=True)
    combined_details_df.to_csv(output_root / ALL_DAYS_DETAILS_FILE, index=False)
    combined_summary_df.to_csv(output_root / ALL_DAYS_SUMMARY_FILE, index=False)
    overall_summary_df.to_csv(output_root / OVERALL_SUMMARY_FILE, index=False)
    return combined_details_df, combined_summary_df, overall_summary_df


def _default_unclear_points() -> list[str]:
    return [
        "The parsed trace files expose a single integer `timestamp`; this runner treats it as Unix microseconds.",
        "The paper constructs STV from RPC message timings, while this dataset provides parsed spans; the implementation uses server-span `duration` as the service response-time proxy.",
        "The paper mentions a whitelist for newly introduced unseen call paths. This benchmark has no explicit deployment whitelist, so unseen call paths are treated as anomalous directly.",
        "If a trace has too few exact HSTV references in the normal set, the localization step falls back to global 3-sigma statistics to keep the benchmark runnable.",
    ]


def run_single_day(script_dir: Path, telemetry_day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_start = perf_counter()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    _, _, traces_dir = get_day_paths(script_dir, telemetry_day)
    fault_runs = discover_fault_runs(script_dir, telemetry_day)
    if EXP_ID:
        filtered_fault_runs = [run for run in fault_runs if run.run_id == EXP_ID]
        if not filtered_fault_runs:
            if len(TELEMETRY_DAYS) == 1:
                raise ValueError(f"EXP_ID not found for {telemetry_day}: {EXP_ID}")
            print(f"[WARN] EXP_ID not found for {telemetry_day}: {EXP_ID}; skipping day")
            return pd.DataFrame(), pd.DataFrame()
        fault_runs = filtered_fault_runs
    normal_runs = discover_normal_runs(script_dir, telemetry_day)
    day_output_dir = (script_dir / OUTPUT_ROOT / telemetry_day).resolve()

    print(
        f"[INFO] telemetry_day={telemetry_day} "
        f"fault_runs={len(fault_runs)} normal_runs={len(normal_runs)}"
    )

    cache = HourlyTraceCache()
    normal_trace_groups: dict[str, list[dict]] = {}
    for index, normal_run in enumerate(normal_runs, start=1):
        trace_df = load_trace_window(cache, traces_dir, normal_run)
        run_groups = build_trace_groups(trace_df)
        for trace_id, spans in run_groups.items():
            normal_trace_groups[f"{normal_run.run_id}:{trace_id}"] = spans
        print(f"[INFO] Loaded normal run {index}/{len(normal_runs)}: {normal_run.run_id} traces={len(run_groups)}")

    if not normal_trace_groups:
        raise ValueError("No normal traces found for the selected telemetry day.")

    train_trace_groups, valid_trace_groups = split_trace_groups(normal_trace_groups)
    call_path_list = build_call_path_vocabulary(train_trace_groups)
    if not call_path_list:
        raise ValueError("No call paths found in normal traces.")
    train_stv_matrix, train_trace_ids, _ = build_stv_matrix(train_trace_groups, call_path_list)
    valid_stv_matrix, valid_trace_ids, _ = build_stv_matrix(valid_trace_groups, call_path_list)
    if train_stv_matrix.size == 0:
        raise ValueError("Normal STV matrix is empty.")
    if valid_stv_matrix.size == 0:
        valid_stv_matrix = train_stv_matrix.copy()
        valid_trace_ids = train_trace_ids.copy()

    means, stds = compute_normalization_stats(train_stv_matrix)
    normalized_train, train_mask = normalize_stv(train_stv_matrix, means, stds)
    normalized_valid, valid_mask = normalize_stv(valid_stv_matrix, means, stds)
    model, training_losses = train_traceanomaly_model(
        normalized_train=normalized_train,
        train_mask=train_mask,
        normalized_valid=normalized_valid,
        valid_mask=valid_mask,
        input_dim=train_stv_matrix.shape[1],
    )
    training_log_likelihoods = estimate_log_likelihoods(model, normalized_train, train_mask) / max(train_stv_matrix.shape[1], 1)
    validation_scores = estimate_log_likelihoods(model, normalized_valid, valid_mask) / max(train_stv_matrix.shape[1], 1)
    validation_score_threshold = float(validation_scores.min()) if len(validation_scores) else float(training_log_likelihoods.min())
    hstv_index = build_hstv_index(train_stv_matrix)

    save_normal_baseline(
        output_dir=day_output_dir / NORMAL_BASELINE_DIR,
        call_path_list=call_path_list,
        means=means,
        stds=stds,
        training_losses=training_losses,
        model=model,
        train_trace_count=len(train_trace_ids),
        validation_trace_count=len(valid_trace_ids),
        normal_run_count=len(normal_runs),
        training_log_likelihoods=training_log_likelihoods,
        validation_scores=validation_scores,
        validation_score_threshold=validation_score_threshold,
        telemetry_day=telemetry_day,
    )

    detail_rows: list[dict] = []
    for index, fault_run in enumerate(fault_runs, start=1):
        run_start = perf_counter()
        print(f"[INFO] [{index}/{len(fault_runs)}] Running {fault_run.run_id}")
        try:
            trace_df = load_trace_window(cache, traces_dir, fault_run)
            trace_groups = build_trace_groups(trace_df)
            fault_stv_matrix, fault_trace_ids, unseen_counts = build_stv_matrix(trace_groups, call_path_list)
            if fault_stv_matrix.size == 0:
                raise ValueError("No valid fault STV traces after call-path alignment.")

            trace_scores_df = score_traces(
                model=model,
                stv_matrix=fault_stv_matrix,
                means=means,
                stds=stds,
                trace_ids=fault_trace_ids,
                unseen_counts=unseen_counts,
                validation_scores=validation_scores,
                validation_score_threshold=validation_score_threshold,
            )
            localization_df = localize_fault_run(
                trace_groups=trace_groups,
                stv_matrix=fault_stv_matrix,
                trace_ids=fault_trace_ids,
                trace_scores_df=trace_scores_df,
                normal_stv_matrix=train_stv_matrix,
                hstv_index=hstv_index,
                call_path_list=call_path_list,
                global_means=means,
                global_stds=stds,
            )
            localization_df = annotate_path_only_topk_fill(localization_df)
            service_ranking_df = aggregate_service_ranking(localization_df)

            runtime_seconds = perf_counter() - run_start
            eval_result = evaluate_topk(service_ranking_df, fault_run.ground_truth_service or "", localization_df)
            run_summary = {
                "telemetry_day": telemetry_day,
                "exp_id": fault_run.run_id,
                "ground_truth_service": fault_run.ground_truth_service,
                "fault_start": str(fault_run.start),
                "fault_end": str(fault_run.end),
                "inject_start": str(fault_run.inject_start) if fault_run.inject_start is not None else None,
                "trace_count": int(len(trace_scores_df)),
                "anomalous_trace_count": int(trace_scores_df["is_anomalous"].sum()) if not trace_scores_df.empty else 0,
                "localized_trace_count": int(len(localization_df)),
                "call_path_count": len(call_path_list),
                "runtime_seconds": runtime_seconds,
                "fault_metadata_path": str(fault_run.metadata_path),
                "p_value_alpha": P_VALUE_ALPHA,
                "anomalous_std_threshold": ANOMALOUS_STD_THRESHOLD,
                "importance_samples": IMPORTANCE_SAMPLES,
                "flow_layers": FLOW_LAYERS,
                "validation_score_threshold": validation_score_threshold,
                "use_secondary_anomalous_dimensions": USE_SECONDARY_ANOMALOUS_DIMENSIONS,
                "secondary_dimension_weight": SECONDARY_DIMENSION_WEIGHT,
                "max_secondary_dimensions_per_trace": MAX_SECONDARY_DIMENSIONS_PER_TRACE,
                "topk_fill_enabled": TOPK_FILL_ENABLED,
                "topk_fill_k": TOPK_FILL_K,
                "topk_fill_mode": TOPK_FILL_MODE,
                **eval_result,
            }
            save_run_outputs(day_output_dir / fault_run.run_id, trace_scores_df, localization_df, service_ranking_df, run_summary)
            detail_rows.append(
                {
                    "telemetry_day": telemetry_day,
                    "exp_id": fault_run.run_id,
                    "ground_truth_service": fault_run.ground_truth_service,
                    "trace_count": int(len(trace_scores_df)),
                    "anomalous_trace_count": int(trace_scores_df["is_anomalous"].sum()) if not trace_scores_df.empty else 0,
                    "localized_trace_count": int(len(localization_df)),
                    "call_path_count": len(call_path_list),
                    "runtime_seconds": runtime_seconds,
                    "use_secondary_anomalous_dimensions": USE_SECONDARY_ANOMALOUS_DIMENSIONS,
                    "secondary_dimension_weight": SECONDARY_DIMENSION_WEIGHT,
                    "max_secondary_dimensions_per_trace": MAX_SECONDARY_DIMENSIONS_PER_TRACE,
                    "topk_fill_enabled": TOPK_FILL_ENABLED,
                    "topk_fill_k": TOPK_FILL_K,
                    "topk_fill_mode": TOPK_FILL_MODE,
                    **eval_result,
                }
            )
            print(
                f"[OK] {fault_run.run_id} top1={eval_result['service_top1_hit']} "
                f"top3={eval_result['service_top3_hit']} top5={eval_result['service_top5_hit']} "
                f"runtime={runtime_seconds:.2f}s anomalous_traces={int(trace_scores_df['is_anomalous'].sum()) if not trace_scores_df.empty else 0}"
            )
        except Exception as exc:
            runtime_seconds = perf_counter() - run_start
            detail_rows.append(
                {
                    "telemetry_day": telemetry_day,
                    "exp_id": fault_run.run_id,
                    "ground_truth_service": fault_run.ground_truth_service,
                    "trace_count": 0,
                    "anomalous_trace_count": 0,
                    "localized_trace_count": 0,
                    "call_path_count": len(call_path_list),
                    "predicted_service_top1": None,
                    "predicted_service_top3": [],
                    "predicted_service_top5": [],
                    "service_top1_hit": False,
                    "service_top3_hit": False,
                    "service_top5_hit": False,
                    "runtime_seconds": runtime_seconds,
                    "use_secondary_anomalous_dimensions": USE_SECONDARY_ANOMALOUS_DIMENSIONS,
                    "secondary_dimension_weight": SECONDARY_DIMENSION_WEIGHT,
                    "max_secondary_dimensions_per_trace": MAX_SECONDARY_DIMENSIONS_PER_TRACE,
                    "topk_fill_enabled": TOPK_FILL_ENABLED,
                    "topk_fill_k": TOPK_FILL_K,
                    "topk_fill_mode": TOPK_FILL_MODE,
                    "error": str(exc),
                }
            )
            print(f"[WARN] {fault_run.run_id} failed: {exc}")

    unclear_points = _default_unclear_points()
    total_runtime_seconds = perf_counter() - total_start
    details_df, summary_df = build_day_outputs(day_output_dir, detail_rows, total_runtime_seconds, unclear_points, telemetry_day)

    if not details_df.empty:
        print(f"\n[DONE] TraceAnomaly day summary: {telemetry_day}")
        print(
            details_df[["service_top1_hit", "service_top3_hit", "service_top5_hit"]]
            .fillna(False)
            .mean()
            .rename(
                {
                    "service_top1_hit": "service_top1_accuracy",
                    "service_top3_hit": "service_top3_accuracy",
                    "service_top5_hit": "service_top5_accuracy",
                }
            )
            .to_frame()
            .T
            .to_string(index=False)
        )
        print(f"[INFO] Total runtime: {total_runtime_seconds:.2f}s")
        print(f"[INFO] Avg runtime per anomaly: {total_runtime_seconds / len(details_df):.2f}s")

    return details_df, summary_df


def main() -> None:
    script_start = perf_counter()
    script_dir = SCRIPT_DIR
    output_root = (script_dir / OUTPUT_ROOT).resolve()

    all_details_dfs: list[pd.DataFrame] = []
    all_summary_dfs: list[pd.DataFrame] = []

    for telemetry_day in TELEMETRY_DAYS:
        details_df, summary_df = run_single_day(script_dir, telemetry_day)
        if not details_df.empty:
            all_details_dfs.append(details_df)
        if not summary_df.empty:
            all_summary_dfs.append(summary_df)

    total_script_runtime_seconds = perf_counter() - script_start
    _, _, overall_summary_df = build_all_days_outputs(
        output_root=output_root,
        all_details_dfs=all_details_dfs,
        all_summary_dfs=all_summary_dfs,
        total_script_runtime_seconds=total_script_runtime_seconds,
    )

    print(f"[DONE] Multi-day details saved: {(output_root / ALL_DAYS_DETAILS_FILE).resolve()}")
    print(f"[DONE] Multi-day summary saved: {(output_root / ALL_DAYS_SUMMARY_FILE).resolve()}")
    print(f"[DONE] Overall summary saved: {(output_root / OVERALL_SUMMARY_FILE).resolve()}")
    if not overall_summary_df.empty:
        print("\n[DONE] TraceAnomaly overall summary")
        print(overall_summary_df.to_string(index=False))
    print(f"[DONE] End-to-end script runtime across {len(TELEMETRY_DAYS)} day(s): {total_script_runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
