"""Species + vocalisation-type classification for a single audio clip.

This is the shared inference core used by BOTH the live installation runtime and the
offline precompute path, so the two can never drift apart.

Pipeline (matches how the models were trained — see CLAUDE.md):
  audio -> BirdNET species detection -> BirdNET embedding (1024-d)
        -> AVES2 embedding (768-d) -> L2-normalise each -> concatenate (1792-d)
        -> per-species projection head -> nearest class prototype (cosine)

Both encoders are loaded ONCE and held warm. The naive path (spinning the BirdNET
embedding stack per call) measured 6.6 s per clip; keeping it warm is what makes live
inference viable at all.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
COMMON_NAMES = {
    "Turdus merula": "blackbird",
    "Erithacus rubecula": "robin",
    "Strix aluco": "tawny_owl",
}


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


@dataclass
class Result:
    species: str | None          # slug, e.g. "blackbird"; None if nothing confident
    species_scientific: str | None
    species_confidence: float
    vocalisation: str | None     # song | call | alarm | juvenile
    vocalisation_confidence: float

    def spoken(self) -> str:
        if not self.species or not self.vocalisation:
            return "unknown"
        return f"{self.species.replace('_', ' ')} {self.vocalisation}"


class Classifier:
    """Holds both encoders and every per-species head in memory."""

    def __init__(self, models_dir: Path, birdnet_binary: str = "birdnet-analyze",
                 aves_model: str = "esp_aves2_sl_beats_bio", device: str = "cpu") -> None:
        self.models_dir = Path(models_dir)
        self.birdnet_binary = birdnet_binary
        self.device = device
        import torch
        self._torch = torch

        from avex import load_model
        self._aves = load_model(aves_model, device=device, return_features_only=True)
        self._aves.eval()

        from .transform_embeddings import _load_checkpoint
        self.heads: dict[str, dict] = {}
        for d in sorted(self.models_dir.iterdir()):
            if not (d / "adapter.pt").exists():
                continue
            _ck, model = _load_checkpoint(d / "adapter.pt")
            model.eval()
            self.heads[d.name] = {
                "model": model,
                "prototypes": _l2(np.load(d / "prototypes.npy")),
                "labels": json.loads((d / "prototype_labels.json").read_text())["labels"],
            }
        if not self.heads:
            raise SystemExit(f"No models found in {self.models_dir}")

    # ---------- stage 1: which bird ----------
    def detect_species(self, wav_path: Path, min_conf: float) -> tuple[str | None, float]:
        from .birdnet_infer import (BirdNETConfig, ensure_birdnet_available,
                                    infer_recordings_with_birdnet, _extract_label, _extract_score)
        ensure_birdnet_available(self.birdnet_binary)
        cfg = BirdNETConfig(binary=self.birdnet_binary, top_k=5)
        preds, _ = infer_recordings_with_birdnet(audio_paths=[wav_path.resolve()], cfg=cfg)
        best: dict[str, float] = {}
        for _rec, windows in preds.items():
            for w in windows:
                for p in w.get("predictions", []):
                    lab = _extract_label(p) or str(p.get("label", "")).strip()
                    sc = _extract_score(p) or float(p.get("score", 0.0) or 0.0)
                    if lab:
                        best[lab] = max(best.get(lab, 0.0), sc)
        # only consider species we actually have a head for
        ranked = sorted(((s, c) for s, c in best.items() if COMMON_NAMES.get(s) in self.heads),
                        key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] < min_conf:
            return None, (ranked[0][1] if ranked else 0.0)
        return ranked[0][0], ranked[0][1]

    # ---------- stage 2: what kind of sound ----------
    # The heads were trained on ~6 s PHRASE UNITS and evaluated by averaging the prototype
    # similarities across all units in a recording. Embedding a whole 25 s clip as one
    # vector is a distribution mismatch and measurably worse (54% vs ~80% on a check of 72
    # clips, 2026-09-08). So: split into phrase-length windows, embed each, and average.
    def _embed_windows(self, y: np.ndarray, window_s: float, hop_s: float) -> np.ndarray:
        import torch, shutil, soundfile as sf
        import pandas as pd
        w, h = int(window_s * SR), int(hop_s * SR)
        starts = list(range(0, max(len(y) - w, 0) + 1, h)) or [0]
        chunks = [y[s:s + w] for s in starts]
        chunks = [c for c in chunks if len(c) >= SR]          # skip stubs under a second

        avs = []
        for c in chunks:
            with torch.no_grad():
                o = self._aves(torch.from_numpy(c.astype(np.float32)).unsqueeze(0))
            avs.append((o.mean(dim=1) if o.ndim == 3 else o).squeeze(0).numpy())

        # one batched BirdNET call for every window — far cheaper than one call per window
        from .embed import birdnet_embedding_array
        with tempfile.TemporaryDirectory(prefix="clf_") as td:
            names = []
            for i, c in enumerate(chunks):
                n = f"w{i:03d}.wav"
                sf.write(Path(td) / n, c, SR)
                names.append(n)
            bn, _meta = birdnet_embedding_array(
                pd.DataFrame([{"event_wav": n} for n in names]), events_dir=Path(td),
                binary=self.birdnet_binary.replace("analyze", "embeddings"))
        return np.stack([np.concatenate([_l2(a), _l2(b)]) for a, b in zip(avs, bn)]).astype(np.float32)

    def classify(self, wav_path: Path, min_species_conf: float = 0.10,
                 window_s: float = 6.0, hop_s: float = 3.0) -> Result:
        import librosa
        sci, conf = self.detect_species(Path(wav_path), min_species_conf)
        if sci is None:
            return Result(None, None, conf, None, 0.0)
        slug = COMMON_NAMES[sci]
        y = librosa.load(str(wav_path), sr=SR, mono=True)[0]
        feats = self._embed_windows(y, window_s, hop_s)
        head = self.heads[slug]
        with self._torch.no_grad():
            z = head["model"](self._torch.tensor(feats)).numpy()
        sims = (_l2(z) @ head["prototypes"].T).mean(axis=0)   # aggregate, as in the evaluation
        i = int(sims.argmax())
        return Result(slug, sci, conf, head["labels"][i], float(sims[i]))
