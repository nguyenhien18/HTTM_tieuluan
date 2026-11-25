"""Train Keras LSTM and GRU on the exact splits used by the saved SimpleRNN notebooks.

Usage: python -X utf8 scripts/source_a06/compare_architectures.py --prepare-only
       python -X utf8 scripts/source_a06/compare_architectures.py --task stock --architecture gru
Existing complete artifacts are kept so an interrupted run can be resumed.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data/raw"
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from rnn_utils import create_behavior_sequences, save_metadata, temporal_split_indices, transition_baseline

RESULTS = ROOT / "runs/sequences/retrained"
REFERENCE = ROOT / "assets/data/a06_results"
SEED = 42
EVENT_TO_ID = {"view": 0, "addtocart": 1, "transaction": 2}
ID_TO_EVENT = {value: key for key, value in EVENT_TO_ID.items()}
ARCHITECTURES = {"SimpleRNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}


def configure_seed():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def artifact_paths(task, architecture):
    prefix = "AMZN" if task == "stock" else "Customer"
    stem = f"{prefix}_Keras_{architecture}"
    return (RESULTS / f"{stem}.keras", RESULTS / f"{stem}_predictions.csv",
            RESULTS / f"{stem}_history.csv", RESULTS / f"{stem}_metrics.csv")


def complete(task, architecture):
    model_path, prediction_path, history_path, metrics_path = artifact_paths(task, architecture)
    return all(path.is_file() for path in (model_path, model_path.with_suffix(".json"),
                                           prediction_path, history_path, metrics_path))


def stock_inputs():
    data_path = DATA_DIR / "AMZN.csv"
    frame = pd.read_csv(data_path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = (frame.dropna(subset=["Date", "Close"]).sort_values("Date")
             .drop_duplicates("Date", keep="last").reset_index(drop=True))
    frame = frame[frame.Date >= "2015-01-01"][["Date", "Close"]].reset_index(drop=True)
    train_end, val_end = int(len(frame) * .70), int(len(frame) * .85)
    scaler = MinMaxScaler().fit(frame.iloc[:train_end][["Close"]])
    scaled = scaler.transform(frame[["Close"]]).astype(np.float32).ravel()

    def windows(start, end):
        indexes = np.arange(start, end)
        x = np.stack([scaled[indexes - offset] for offset in range(30, 0, -1)], axis=1)[..., None]
        return x, scaled[indexes]

    X_train, y_train = windows(30, train_end)
    X_val, y_val = windows(train_end, val_end)
    X_test, y_test = windows(val_end, len(frame))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    previous = frame.Close.iloc[val_end - 1:-1].to_numpy()
    dates = frame.Date.iloc[val_end:].reset_index(drop=True)
    reference = pd.read_csv(REFERENCE / "AMZN_Keras_predictions.csv")
    assert len(X_train) == 1542 and len(X_val) == 337 and len(X_test) == 338
    assert dates.dt.strftime("%Y-%m-%d").tolist() == reference.Date.tolist()
    np.testing.assert_allclose(actual, reference.Actual_Close, atol=1e-4)
    split_dates = {"train_end": str(frame.Date.iloc[train_end - 1]),
                   "val_start": str(frame.Date.iloc[train_end]),
                   "val_end": str(frame.Date.iloc[val_end - 1]),
                   "test_start": str(frame.Date.iloc[val_end])}
    return locals()


def train_stock(architecture, data):
    model_path, prediction_path, history_path, metrics_path = artifact_paths("stock", architecture)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, scaler = data["X_test"], data["scaler"]
    best_val_mse = float("inf")
    selected = None
    search = []
    for units, learning_rate in [(32, .001), (50, .001), (50, .0005)]:
        tf.keras.backend.clear_session()
        configure_seed()
        model = tf.keras.Sequential([
            layers.Input(shape=(30, 1)),
            ARCHITECTURES[architecture](units),
            layers.Dense(1),
        ], name=f"Stock{architecture}")
        model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
                      loss="mse", metrics=["mae"])
        print(f"{architecture} stock: units={units}, lr={learning_rate}", flush=True)
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=80,
            batch_size=32, shuffle=True,
            callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                       ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=4, min_lr=1e-5)],
            verbose=0)
        prediction = model.predict(X_val, verbose=0).ravel()
        score = float(np.mean((prediction - y_val) ** 2))
        if not np.isfinite(score):
            raise RuntimeError("Non-finite stock validation MSE")
        row = {"units": units, "learning_rate": learning_rate, "val_mse": score,
               "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
               "epochs": len(history.history["loss"])}
        search.append(row)
        print(row, flush=True)
        if score < best_val_mse:
            best_val_mse = score
            selected = row
            selected_history = pd.DataFrame(history.history)
            model.save(model_path)
        del model, history
        gc.collect()

    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    predicted_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    predicted = scaler.inverse_transform(predicted_scaled).ravel()
    actual, previous, dates = data["actual"], data["previous"], data["dates"]
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    direction = float(np.mean(np.sign(np.where(abs(actual - previous) <= 1e-4, 0, actual - previous))
                              == np.sign(np.where(abs(predicted - previous) <= 1e-4, 0, predicted - previous))))
    baseline_mae = float(mean_absolute_error(actual, previous))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, previous)))
    pd.DataFrame({"Date": dates.dt.strftime("%Y-%m-%d"), "Previous_Close": previous,
                  "Actual_Close": actual, "RNN_Predicted_Close": predicted,
                  "Baseline_Predicted_Close": previous,
                  "RNN_Absolute_Error": abs(actual - predicted)}).to_csv(prediction_path, index=False)
    pd.DataFrame([{"Model": architecture, "MAE": mae, "MSE": rmse ** 2, "RMSE": rmse},
                  {"Model": "Naive baseline", "MAE": baseline_mae,
                   "MSE": baseline_rmse ** 2, "RMSE": baseline_rmse}]).to_csv(metrics_path, index=False)
    selected_history.to_csv(history_path, index=False)
    pd.DataFrame(search).to_csv(RESULTS / f"AMZN_Keras_{architecture}_validation_search.csv", index=False)
    save_metadata(model_path.with_suffix(".json"), task="stock", framework="keras", seed=SEED,
                  seq_length=30, dataset=DATA_DIR / "AMZN.csv", architecture=architecture.lower(),
                  hidden_size=selected["units"], learning_rate=selected["learning_rate"],
                  best_epoch=selected["best_epoch"], selected_val_mse=best_val_mse,
                  selection_metric="val_mse", train_samples=len(X_train),
                  validation_samples=len(X_val), test_samples=len(X_test),
                  scaler={"scale": scaler.scale_.tolist(), "min": scaler.min_.tolist(),
                          "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist()},
                  split_dates=data["split_dates"],
                  test_metrics={"mae": mae, "rmse": rmse, "baseline_rmse": baseline_rmse,
                                "direction_accuracy": direction})
    print(f"{architecture} stock: test RMSE={rmse:.4f}, baseline={baseline_rmse:.4f}", flush=True)


def customer_inputs():
    data_path = DATA_DIR / "events.csv"
    print("Loading and preparing customer events...", flush=True)
    frame = pd.read_csv(data_path).drop_duplicates()
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", errors="coerce")
    frame["event"] = frame["event"].astype(str).str.strip().str.lower()
    frame = frame[frame.datetime.notna() & frame.visitorid.notna() & frame.event.isin(EVENT_TO_ID)].copy()
    frame["visitorid"] = pd.to_numeric(frame["visitorid"], errors="coerce")
    frame["itemid"] = pd.to_numeric(frame["itemid"], errors="coerce")
    frame = frame.dropna(subset=["visitorid", "itemid"])
    frame["visitorid"] = frame["visitorid"].astype("int64")
    frame["itemid"] = frame["itemid"].astype("int64")
    frame = frame.sort_values(["visitorid", "datetime", "timestamp"]).reset_index(drop=True)
    counts = frame.groupby("visitorid").size()
    frame = frame[frame.visitorid.isin(counts[counts >= 6].index)].copy()
    frame["event_id"] = frame.event.map(EVENT_TO_ID).astype("int32")
    X_all, y_all, time_all, visitor_all = create_behavior_sequences(frame, 5)
    del frame
    gc.collect()
    sequences = pd.DataFrame({"idx": np.arange(len(y_all)), "target_time": time_all,
                              "target": y_all, "visitorid": visitor_all})
    sequences = sequences.sort_values("target_time", kind="stable").reset_index(drop=True)
    if len(sequences) > 200_000:
        positions = np.linspace(0, len(sequences) - 1, 200_000, dtype=int)
        sequences = sequences.iloc[positions].reset_index(drop=True)
    selected = sequences.idx.to_numpy()
    X, y = X_all[selected], y_all[selected]
    times = sequences.target_time.reset_index(drop=True)
    del X_all, y_all, time_all, visitor_all, sequences
    gc.collect()
    train_end, val_end = temporal_split_indices(times)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    test_times = times.iloc[val_end:].reset_index(drop=True)
    assert (len(X_train), len(X_val), len(X_test)) == (140000, 30000, 30000)
    reference = pd.read_csv(REFERENCE / "Customer_Keras_predictions.csv")
    assert [ID_TO_EVENT[int(v)] for v in y_test] == reference.actual_event.tolist()
    assert pd.to_datetime(reference.target_time).tolist() == test_times.tolist()
    split_dates = {"train_end": str(times.iloc[train_end - 1]),
                   "val_start": str(times.iloc[train_end]),
                   "val_end": str(times.iloc[val_end - 1]),
                   "test_start": str(times.iloc[val_end])}
    print(f"Customer split verified: {len(X_train)}/{len(X_val)}/{len(X_test)}", flush=True)
    return locals()


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.X_val, batch_size=1024, verbose=0).argmax(axis=1)
        logs["val_macro_f1"] = f1_score(self.y_val, predicted, labels=[0, 1, 2],
                                        average="macro", zero_division=0)
        print(f"  epoch {epoch + 1}: val_macro_f1={logs['val_macro_f1']:.4f}", flush=True)


def train_customer(architecture, data):
    model_path, prediction_path, history_path, metrics_path = artifact_paths("customer", architecture)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    val_weight = np.asarray([class_weight[int(label)] for label in y_val], dtype=np.float32)
    tf.keras.backend.clear_session()
    configure_seed()
    model = tf.keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Embedding(input_dim=3, output_dim=8),
        ARCHITECTURES[architecture](32),
        layers.Dense(3, activation="softmax"),
    ], name=f"Customer{architecture}")
    model.compile(optimizer=Adam(learning_rate=.001, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"{architecture} customer: training", flush=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val, val_weight),
                        epochs=30, batch_size=256, shuffle=True, class_weight=class_weight,
                        callbacks=[ValidationMacroF1(X_val, y_val),
                                   EarlyStopping(monitor="val_macro_f1", mode="max", patience=6,
                                                 restore_best_weights=True),
                                   ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=.5,
                                                     patience=3, min_lr=1e-5)], verbose=2)
    probabilities = model.predict(X_test, batch_size=1024, verbose=0)
    predicted = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(y_test, predicted))
    macro_f1 = float(f1_score(y_test, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, predicted, labels=[0, 1, 2], average="weighted", zero_division=0))
    pd.DataFrame({"target_time": data["test_times"],
                  "actual_event": [ID_TO_EVENT[int(v)] for v in y_test],
                  "predicted_event": [ID_TO_EVENT[int(v)] for v in predicted],
                  "prob_view": probabilities[:, 0], "prob_addtocart": probabilities[:, 1],
                  "prob_transaction": probabilities[:, 2]}).to_csv(prediction_path, index=False)
    baseline = np.full(len(y_test), np.bincount(y_train).argmax(), dtype=np.int32)
    comparison = [{"Model": architecture, "Accuracy": accuracy, "Macro_F1": macro_f1}]
    for name, output in [("Majority Baseline", baseline), ("Last event", X_test[:, -1]),
                         ("Markov", transition_baseline(X_train, y_train, X_test)),
                         ("Balanced Markov", transition_baseline(X_train, y_train, X_test, balance=True))]:
        comparison.append({"Model": name, "Accuracy": accuracy_score(y_test, output),
                           "Macro_F1": f1_score(y_test, output, labels=[0, 1, 2],
                                                average="macro", zero_division=0)})
    pd.DataFrame(comparison).to_csv(metrics_path, index=False)
    report = classification_report(y_test, predicted, labels=[0, 1, 2],
                                   target_names=list(EVENT_TO_ID), zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(RESULTS / f"Customer_Keras_{architecture}_classification_report.csv")
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    model.save(model_path)
    save_metadata(model_path.with_suffix(".json"), task="customer", framework="keras", seed=SEED,
                  seq_length=5, dataset=DATA_DIR / "events.csv",
                  architecture=f"embedding_{architecture.lower()}", hidden_size=32,
                  embed_dim=8, learning_rate=.001, selection_metric="val_macro_f1",
                  best_epoch=int(np.argmax(history.history["val_macro_f1"])) + 1,
                  train_samples=len(X_train), validation_samples=len(X_val),
                  test_samples=len(X_test), event_to_id=EVENT_TO_ID,
                  probability_calibrated=False, strict_past_events=True,
                  split_dates=data["split_dates"],
                  test_metrics={"accuracy": accuracy, "macro_f1": macro_f1,
                                "weighted_f1": weighted_f1})
    print(f"{architecture} customer: test Macro F1={macro_f1:.4f}, Accuracy={accuracy:.4f}", flush=True)


def main():
    global DATA_DIR, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["stock", "customer", "all"], default="all")
    parser.add_argument("--architecture", choices=["rnn", "lstm", "gru", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    DATA_DIR=args.data_dir.resolve();RESULTS=args.output_dir.resolve()
    RESULTS.mkdir(parents=True,exist_ok=True)
    architectures = [("SimpleRNN" if args.architecture=="rnn" else args.architecture.upper())] if args.architecture != "all" else list(ARCHITECTURES)
    for task, prepare, train in [("stock", stock_inputs, train_stock),
                                 ("customer", customer_inputs, train_customer)]:
        if args.task not in ("all", task):
            continue
        pending = [architecture for architecture in architectures if not complete(task, architecture)]
        if not pending and not args.prepare_only:
            print(f"{task}: all requested artifacts exist; skipping", flush=True)
            continue
        data = prepare()
        if args.prepare_only:
            report=dict(task=task,train=len(data['X_train']),validation=len(data['X_val']),test=len(data['X_test']),split_dates=data['split_dates'],reference_labels_and_dates_match=True)
            (RESULTS/f'{task}_preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(report,flush=True)
            del data;gc.collect();continue
        for architecture in pending:
            train(architecture, data)
        del data
        gc.collect()


if __name__ == "__main__":
    main()
