"""Frozen shared-CBC inference; no fitting or patient-data persistence."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

FEATURE_ORDER = ["LBXWBCSI", "LBXLYPCT", "LBXNEPCT", "LBXMOPCT", "LBXRBCSI", "NLR", "LMR"]
INPUT_NAMES = ["wbc", "rbc", "neutrophils", "lymphocytes", "monocytes"]


def load_bundle(directory=None):
    directory = Path(directory) if directory else Path(__file__).resolve().parent / "bundle"
    config = json.loads((directory / "config.json").read_text())
    path = directory / "model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["model_sha256"]:
        raise RuntimeError("Model checksum does not match deployment configuration")
    if config["features"] != FEATURE_ORDER:
        raise RuntimeError("Unsupported model feature order")
    model = joblib.load(path)
    if model.n_features_in_ != len(FEATURE_ORDER):
        raise RuntimeError("Model input shape does not match deployment configuration")
    return model, config


def make_features(values):
    if set(values) != set(INPUT_NAMES):
        raise ValueError("Provide exactly the five CBC measurements")
    clean = {}
    for key in INPUT_NAMES:
        if isinstance(values[key], bool) or values[key] is None:
            raise ValueError("Complete all five numerical measurements")
        try:
            value = float(values[key])
        except (ValueError, TypeError):
            raise ValueError("Complete all five numerical measurements") from None
        if not np.isfinite(value):
            raise ValueError("Measurements must be finite numbers")
        clean[key] = value
    if clean["wbc"] <= 0 or clean["rbc"] <= 0:
        raise ValueError("WBC and RBC counts must be greater than zero")
    differential = [clean[key] for key in ["neutrophils", "lymphocytes", "monocytes"]]
    if any(value < 0 or value > 100 for value in differential):
        raise ValueError("Differential percentages must be between 0 and 100")
    if sum(differential) > 100 + 1e-6:
        raise ValueError("The three differential percentages cannot total more than 100")
    # Match pandas .round(0), used during precision-harmonized training.
    neut, lymph, mono = np.round(differential, 0)
    nlr = neut / lymph if lymph else np.nan
    lmr = lymph / mono if mono else np.nan
    row = np.array([[clean["wbc"], lymph, neut, mono, clean["rbc"], nlr, lmr]], dtype=float)
    return row


def predict(model, config, values):
    row = make_features(values)
    score = float(model.predict_proba(row)[0, 1])
    return {"score": score, "cutoff": float(config["threshold"]),
            "above_cutoff": score >= config["threshold"],
            "undefined_ratios_imputed": bool(np.isnan(row).any())}
