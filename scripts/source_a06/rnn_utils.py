"""Shared, framework-independent preprocessing and experiment metadata."""
from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd


def project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data/AMZN.csv").is_file() and (candidate / "data/events.csv").is_file():
            return candidate
    raise FileNotFoundError("Run from the A06_RNN directory or its notebooks directory.")


def create_behavior_sequences(data, seq_length=5):
    """Only past events from the same visitor; exclude ambiguous timestamp ties.

    Allocate arrays once instead of keeping millions of small Python arrays alive.
    No label-dependent sampling or fitting occurs here.
    """
    if seq_length < 1:
        raise ValueError("seq_length must be positive")
    data = data.sort_values(["visitorid", "datetime"], kind="stable")
    sizes = data.groupby("visitorid", sort=False).size()
    capacity = int((sizes - seq_length).clip(lower=0).sum())
    X = np.empty((capacity, seq_length), dtype=np.int32)
    y = np.empty(capacity, dtype=np.int32)
    times = np.empty(capacity, dtype="datetime64[ns]")
    visitors = np.empty(capacity, dtype=np.int64)
    offset = 0
    for visitor, group in data.groupby("visitorid", sort=False):
        if len(group) <= seq_length:
            continue
        events = group["event_id"].to_numpy(dtype=np.int32)
        stamps = group["datetime"].to_numpy(dtype="datetime64[ns]")
        windows = np.lib.stride_tricks.sliding_window_view(events, seq_length + 1)
        valid = stamps[seq_length - 1:-1] < stamps[seq_length:]
        count = int(valid.sum())
        end = offset + count
        X[offset:end] = windows[valid, :-1]
        y[offset:end] = windows[valid, -1]
        times[offset:end] = stamps[seq_length:][valid]
        visitors[offset:end] = visitor
        offset = end
    if offset == 0:
        raise ValueError("No valid sequences: need enough strictly earlier events per visitor.")
    return X[:offset], y[:offset], pd.to_datetime(times[:offset]), visitors[:offset]


def temporal_split_indices(times, train_fraction=0.70, val_fraction=0.15):
    """Move boundaries before ties so equal target timestamps never cross splits."""
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("Split fractions must be positive and leave a nonempty test fraction")
    values = np.asarray(times, dtype="datetime64[ns]")
    if len(values) < 3 or np.isnat(values).any() or (values[1:] < values[:-1]).any():
        raise ValueError("Need at least three sorted, valid target timestamps")
    train = int(len(values) * train_fraction)
    val = int(len(values) * (train_fraction + val_fraction))
    train = int(np.searchsorted(values, values[train], side="left"))
    val = int(np.searchsorted(values, values[val], side="left"))
    if not 0 < train < val < len(values):
        raise ValueError("Not enough distinct timestamps for three nonempty splits")
    return train, val


def transition_baseline(X_train, y_train, X, n_classes=3, balance=False):
    """First-order Markov baseline fitted exclusively on train, with Laplace smoothing."""
    counts = np.ones((n_classes, n_classes), dtype=np.float64)
    np.add.at(counts, (X_train[:, -1], y_train), 1)
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    if balance:
        frequencies = np.bincount(y_train, minlength=n_classes).clip(min=1)
        probabilities /= frequencies / frequencies.sum()
    return probabilities[X[:, -1]].argmax(axis=1)


def save_metadata(path, *, task, framework, seed, seq_length, dataset, **extra):
    digest = hashlib.sha256()
    with Path(dataset).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = dict(schema_version=1, task=task, framework=framework, seed=seed,
                    seq_length=seq_length, dataset_sha256=digest.hexdigest(),
                    package_versions=versions, **extra)
    Path(path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return metadata
