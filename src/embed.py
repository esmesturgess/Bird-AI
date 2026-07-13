from __future__ import annotations

import csv
import json
import dataclasses
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


EmbeddingBackend = Literal["birdnet", "mel_time", "mel_stats", "avex_effnet_bio"]
FallbackBackend = Literal["none", "mel_time", "mel_stats"]

AVEX_EFFNET_BIO_MODEL = "esp_aves2_effnetb0_bio"
AVEX_EFFNET_BIO_SAMPLE_RATE = 16000
AVEX_EFFNET_BIO_TARGET_SECONDS = 10.0


def _compute_mel_db(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        fmin=200,
        fmax=min(12000, sr // 2),
        power=2.0,
    )
    return librosa.power_to_db(np.maximum(mel, 1e-12), ref=np.max)


def _resample_time_axis(x: np.ndarray, target_frames: int) -> np.ndarray:
    if x.shape[1] == target_frames:
        return x
    if x.shape[1] <= 1:
        return np.repeat(x, repeats=target_frames, axis=1)
    src = np.linspace(0.0, 1.0, x.shape[1], endpoint=True)
    dst = np.linspace(0.0, 1.0, target_frames, endpoint=True)
    out = np.empty((x.shape[0], target_frames), dtype=np.float32)
    for band in range(x.shape[0]):
        out[band] = np.interp(dst, src, x[band]).astype(np.float32)
    return out


def mel_stats_embedding(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    mel_db = _compute_mel_db(y=y, sr=sr, n_mels=n_mels)
    mean = np.mean(mel_db, axis=1)
    std = np.std(mel_db, axis=1)
    delta = librosa.feature.delta(mel_db)
    delta_mean = np.mean(delta, axis=1)
    return np.concatenate([mean, std, delta_mean]).astype(np.float32)


def mel_time_embedding(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_frames: int = 96,
) -> np.ndarray:
    mel_db = _compute_mel_db(y=y, sr=sr, n_mels=n_mels)
    mel_resized = _resample_time_axis(mel_db, target_frames=n_frames)
    mean = float(np.mean(mel_resized))
    std = float(np.std(mel_resized))
    mel_norm = (mel_resized - mean) / (std + 1e-6)
    return mel_norm.reshape(-1).astype(np.float32)


def _expected_dim(backend: EmbeddingBackend, n_mels: int, mel_frames: int) -> int | None:
    if backend == "mel_stats":
        return n_mels * 3
    if backend == "mel_time":
        return n_mels * mel_frames
    return None


def _embed_one(
    y: np.ndarray,
    sr: int,
    backend: Literal["mel_time", "mel_stats"],
    n_mels: int,
    mel_frames: int,
) -> np.ndarray:
    if backend == "mel_stats":
        return mel_stats_embedding(y=y, sr=sr, n_mels=n_mels)
    return mel_time_embedding(y=y, sr=sr, n_mels=n_mels, n_frames=mel_frames)


def _center_pad_or_crop(y: np.ndarray, target_samples: int) -> np.ndarray:
    if y.size == target_samples:
        return y.astype(np.float32, copy=False)
    if y.size > target_samples:
        start = max(0, (y.size - target_samples) // 2)
        return y[start : start + target_samples].astype(np.float32, copy=False)
    pad_total = max(0, target_samples - y.size)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y.astype(np.float32, copy=False), (pad_left, pad_right), mode="constant")


def avex_effnet_bio_embedding_array(
    clips_df: pd.DataFrame,
    events_dir: Path,
    batch_size: int = 4,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    import avex

    model_name = AVEX_EFFNET_BIO_MODEL
    checkpoint_path = avex.get_checkpoint_path(model_name)
    model = avex.load_model(
        model_name,
        device=device,
        checkpoint_path=checkpoint_path,
        return_features_only=True,
    )
    model.eval()

    sample_rate = AVEX_EFFNET_BIO_SAMPLE_RATE
    target_samples = int(round(sample_rate * AVEX_EFFNET_BIO_TARGET_SECONDS))
    vectors: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not batch:
            return
        x = torch.stack(batch, dim=0).to(device)
        with torch.no_grad():
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("embedding", next(iter(out.values())))
        if out.ndim > 2:
            out = out.mean(dim=tuple(range(2, out.ndim)))
        vectors.extend(out.detach().cpu().numpy().astype(np.float32))
        batch.clear()

    for _, row in tqdm(clips_df.iterrows(), total=len(clips_df), desc="AVEX embeddings"):
        clip_path = events_dir / str(row["event_wav"])
        y, _ = librosa.load(clip_path.as_posix(), sr=sample_rate, mono=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        y = _center_pad_or_crop(y, target_samples=target_samples)
        batch.append(torch.from_numpy(y))
        if len(batch) >= int(batch_size):
            flush_batch()
    flush_batch()

    emb_array = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
    extras = {
        "backend": "avex_effnet_bio",
        "embedding_dim": int(emb_array.shape[1]) if emb_array.ndim == 2 else 0,
        "avex_model": model_name,
        "avex_checkpoint_path": checkpoint_path or "",
        "avex_sample_rate": sample_rate,
        "avex_target_seconds": AVEX_EFFNET_BIO_TARGET_SECONDS,
        "avex_pooling": "global_mean_over_feature_map",
    }
    return emb_array.astype(np.float32), extras


def _normalize_path_key(value: str) -> str:
    return value.strip().replace("\\", "/")


def _parse_birdnet_embedding_row(row: dict[str, str]) -> tuple[str, np.ndarray] | None:
    norm = {str(k).strip().lower().replace(" ", "_"): (v or "").strip() for k, v in row.items()}
    file_path = norm.get("file_path") or norm.get("file") or ""
    embedding_str = norm.get("embedding") or ""
    if not file_path or not embedding_str:
        return None
    try:
        vec = np.asarray([float(v) for v in embedding_str.split(",")], dtype=np.float32)
    except ValueError:
        return None
    if vec.size == 0:
        return None
    return _normalize_path_key(file_path), vec


def _path_keys(path: Path, events_dir: Path) -> list[str]:
    resolved = _normalize_path_key(path.resolve().as_posix())
    rel = _normalize_path_key(path.relative_to(events_dir).as_posix()) if path.is_relative_to(events_dir) else path.name
    name = _normalize_path_key(path.name)
    return [resolved, rel, name]


def _patch_perch_hoplite_compat() -> None:
    from ml_collections import config_dict
    from perch_hoplite.db import interface as hoplite_interface  # type: ignore
    from perch_hoplite.db import sqlite_usearch_impl  # type: ignore

    if hasattr(sqlite_usearch_impl, "SQLiteUSearchDB") and not hasattr(sqlite_usearch_impl, "SQLiteUsearchDB"):
        setattr(sqlite_usearch_impl, "SQLiteUsearchDB", getattr(sqlite_usearch_impl, "SQLiteUSearchDB"))

    if not hasattr(hoplite_interface, "EmbeddingSource"):
        @dataclasses.dataclass
        class EmbeddingSource:
            dataset_name: str
            source_id: str
            offsets: np.ndarray

        setattr(hoplite_interface, "EmbeddingSource", EmbeddingSource)

    cls = getattr(sqlite_usearch_impl, "SQLiteUSearchDB", None)
    if cls is None:
        return

    if not hasattr(cls, "get_embedding_ids"):
        def get_embedding_ids(self) -> list[int]:
            return [int(window.id) for window in self.get_all_windows()]

        setattr(cls, "get_embedding_ids", get_embedding_ids)

    if not hasattr(cls, "get_embedding_source"):
        def get_embedding_source(self, embedding_id: int):
            window = self.get_window(int(embedding_id))
            recording = self.get_recording(int(window.recording_id))
            embedding_source_cls = getattr(hoplite_interface, "EmbeddingSource")
            return embedding_source_cls(
                dataset_name="birdnet_analyzer_dataset",
                source_id=str(recording.filename),
                offsets=np.asarray(window.offsets, dtype=float),
            )

        setattr(cls, "get_embedding_source", get_embedding_source)

    if not hasattr(cls, "get_embeddings_by_source"):
        def get_embeddings_by_source(self, dataset_name: str, source_id: str, offsets: np.ndarray) -> np.ndarray:
            del dataset_name
            recordings = self.get_all_recordings(filter=config_dict.create(eq=dict(filename=str(source_id))))
            if not recordings:
                return np.zeros((0, int(self.get_embedding_dim())), dtype=np.float32)
            windows = self.get_all_windows(
                filter=config_dict.create(
                    eq=dict(recording_id=int(recordings[0].id)),
                    approx=dict(offsets=np.asarray(offsets, dtype=float).tolist()),
                )
            )
            if not windows:
                return np.zeros((0, int(self.get_embedding_dim())), dtype=np.float32)
            vectors = [self.get_embedding(int(window.id)).astype(np.float32) for window in windows]
            return np.vstack(vectors) if vectors else np.zeros((0, int(self.get_embedding_dim())), dtype=np.float32)

        setattr(cls, "get_embeddings_by_source", get_embeddings_by_source)

    if not hasattr(cls, "insert_embedding"):
        def insert_embedding(self, embedding: np.ndarray, embedding_source) -> int:
            source_id = str(getattr(embedding_source, "source_id"))
            offsets = np.asarray(getattr(embedding_source, "offsets"), dtype=float).tolist()
            recordings = self.get_all_recordings(filter=config_dict.create(eq=dict(filename=source_id)))
            if recordings:
                recording_id = int(recordings[0].id)
            else:
                recording_id = int(self.insert_recording(filename=source_id))
            return int(
                self.insert_window(
                    recording_id=recording_id,
                    offsets=offsets,
                    embedding=np.asarray(embedding, dtype=np.float32),
                    handle_duplicates="skip",
                )
            )

        setattr(cls, "insert_embedding", insert_embedding)


def birdnet_embedding_array(
    clips_df: pd.DataFrame,
    events_dir: Path,
    binary: str,
    timeout_s: int = 600,
) -> tuple[np.ndarray, dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="birdnet_embed_") as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "birdnet_embeddings.sqlite"
        csv_out = tmp_dir / "birdnet_embeddings.csv"
        python_err = ""
        binary_err = ""

        try:
            _patch_perch_hoplite_compat()
            from birdnet_analyzer.embeddings.core import embeddings as run_embeddings

            run_embeddings(
                audio_input=events_dir.as_posix(),
                database=db_path.as_posix(),
                overlap=0.0,
                audio_speed=1.0,
                fmin=0,
                fmax=15000,
                threads=1,
                batch_size=1,
                file_output=csv_out.as_posix(),
            )
        except Exception as exc:
            python_err = f"{type(exc).__name__}: {exc}"

        if not csv_out.exists():
            resolved_bin = shutil.which(binary)
            if resolved_bin:
                cmd = [
                    resolved_bin,
                    "-i",
                    events_dir.as_posix(),
                    "-db",
                    db_path.as_posix(),
                    "-t",
                    "1",
                    "--file_output",
                    csv_out.as_posix(),
                    "--overlap",
                    "0.0",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip()
                    stdout = (proc.stdout or "").strip()
                    binary_err = stderr.splitlines()[-1] if stderr else (stdout.splitlines()[-1] if stdout else "unknown error")
                elif not csv_out.exists():
                    binary_err = "birdnet-embeddings did not produce --file_output CSV."
            else:
                binary_err = f"binary '{binary}' was not found in PATH."

        if not csv_out.exists():
            raise RuntimeError(f"BirdNET embeddings failed. python_api={python_err}; binary={binary_err}")

        rows: list[dict[str, str]] = []
        with csv_out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: str(v) if v is not None else "" for k, v in row.items()})

    vectors_by_key: dict[str, list[np.ndarray]] = {}
    for row in rows:
        parsed = _parse_birdnet_embedding_row(row)
        if parsed is None:
            continue
        file_key, vec = parsed
        vectors_by_key.setdefault(file_key, []).append(vec)
        vectors_by_key.setdefault(Path(file_key).name, []).append(vec)

    if not vectors_by_key:
        raise RuntimeError(
            "birdnet-embeddings returned no parseable vectors. "
            "If you see ml_collections errors, install it in the venv."
        )

    first_vec = next(iter(vectors_by_key.values()))[0]
    emb_dim = int(first_vec.size)
    embeddings = np.zeros((len(clips_df), emb_dim), dtype=np.float32)
    missing = 0

    for i, row in clips_df.iterrows():
        clip_path = events_dir / str(row["event_wav"])
        candidates = _path_keys(clip_path, events_dir=events_dir)
        found: list[np.ndarray] = []
        for key in candidates:
            if key in vectors_by_key:
                found = vectors_by_key[key]
                break
        if not found:
            missing += 1
            continue
        clip_matrix = np.vstack(found)
        embeddings[i] = np.mean(clip_matrix, axis=0).astype(np.float32)

    return embeddings, {"backend": "birdnet", "embedding_dim": emb_dim, "missing_clips": missing}


def generate_embeddings(
    clips_df: pd.DataFrame,
    events_dir: Path,
    sr: int,
    out_path: Path,
    backend: EmbeddingBackend = "mel_time",
    n_mels: int = 64,
    mel_frames: int = 96,
    birdnet_embeddings_binary: str = "birdnet-embeddings",
    fallback_backend: FallbackBackend = "mel_time",
    force: bool = False,
) -> np.ndarray:
    meta_path = out_path.with_name("embedding_meta.json")
    index_path = out_path.with_name("embedding_index.csv")

    if out_path.exists() and meta_path.exists() and not force:
        emb = np.load(out_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        requested = str(meta.get("requested_backend", meta.get("backend", "")))
        if emb.shape[0] == len(clips_df) and requested == backend:
            return emb

    requested_backend: EmbeddingBackend = backend
    effective_backend: EmbeddingBackend = backend
    fallback_reason = ""
    extras: dict[str, object] = {}

    if backend == "birdnet":
        try:
            emb_array, extras = birdnet_embedding_array(
                clips_df=clips_df,
                events_dir=events_dir,
                binary=birdnet_embeddings_binary,
            )
        except Exception as exc:
            if fallback_backend == "none":
                raise
            effective_backend = fallback_backend
            fallback_reason = f"{type(exc).__name__}: {exc}"
            print(f"[embed] BirdNET embedding failed; falling back to '{effective_backend}': {fallback_reason}")
            embeddings: list[np.ndarray] = []
            for _, row in tqdm(clips_df.iterrows(), total=len(clips_df), desc="Embeddings"):
                clip_path = events_dir / row["event_wav"]
                y, _ = librosa.load(clip_path.as_posix(), sr=sr, mono=True)
                embeddings.append(
                    _embed_one(y=y, sr=sr, backend=effective_backend, n_mels=n_mels, mel_frames=mel_frames)
                )
            expected_dim = _expected_dim(backend=effective_backend, n_mels=n_mels, mel_frames=mel_frames) or 0
            emb_array = (
                np.vstack(embeddings) if embeddings else np.zeros((0, expected_dim), dtype=np.float32)
            )
    elif backend == "avex_effnet_bio":
        emb_array, extras = avex_effnet_bio_embedding_array(
            clips_df=clips_df,
            events_dir=events_dir,
        )
    else:
        embeddings = []
        for _, row in tqdm(clips_df.iterrows(), total=len(clips_df), desc="Embeddings"):
            clip_path = events_dir / row["event_wav"]
            y, _ = librosa.load(clip_path.as_posix(), sr=sr, mono=True)
            embeddings.append(_embed_one(y=y, sr=sr, backend=backend, n_mels=n_mels, mel_frames=mel_frames))
        expected_dim = _expected_dim(backend=backend, n_mels=n_mels, mel_frames=mel_frames) or 0
        emb_array = np.vstack(embeddings) if embeddings else np.zeros((0, expected_dim), dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb_array.astype(np.float32))

    index_df = pd.DataFrame({"clip_id": clips_df["clip_id"].to_numpy(), "embedding_row": np.arange(len(clips_df))})
    index_df.to_csv(index_path, index=False)

    meta_payload = {
        "requested_backend": requested_backend,
        "backend": effective_backend,
        "sample_rate": sr,
        "n_mels": n_mels,
        "mel_frames": mel_frames,
        "embedding_dim": int(emb_array.shape[1]) if emb_array.ndim == 2 else 0,
        "n_clips": int(len(clips_df)),
        "birdnet_embeddings_binary": birdnet_embeddings_binary if requested_backend == "birdnet" else "",
        "fallback_backend": fallback_backend,
        "fallback_reason": fallback_reason,
        **extras,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")
    return emb_array
