import argparse, hashlib, json, platform
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

FEATURES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
CORE_PACKAGES = ["numpy", "scipy", "scikit-learn", "pandas", "joblib"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data(path):
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ["label"]:
        raise ValueError("Wrong columns or column order")
    values = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Measurements must be finite and positive")
    if not frame["label"].isin([0, 1, 2]).all():
        raise ValueError("Unknown label")
    return frame[FEATURES], frame["label"]

def run(data_dir, config_path, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    cfg = json.loads(Path(config_path).read_text())
    expected_keys = {"max_depth", "min_samples_leaf", "random_state"}
    if set(cfg) != expected_keys:
        raise ValueError("Config must contain exactly: " + str(sorted(expected_keys)))
    if cfg["max_depth"] is not None and (type(cfg["max_depth"]) is not int or cfg["max_depth"] < 1):
        raise ValueError("max_depth must be a positive integer or null")
    if type(cfg["min_samples_leaf"]) is not int or cfg["min_samples_leaf"] < 1:
        raise ValueError("min_samples_leaf must be a positive integer")
    if type(cfg["random_state"]) is not int:
        raise ValueError("random_state must be an integer")
    X_train, y_train = load_data(data_dir / "train.csv")
    X_val, y_val = load_data(data_dir / "validation.csv")
    model = DecisionTreeClassifier(**cfg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    metrics = {"train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
               "val_accuracy": float(accuracy_score(y_val, predictions)),
               "val_macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
               "node_count": int(model.tree_.node_count)}
    evidence = {"config": cfg, "code_sha256": sha(__file__),
                "train_sha256": sha(data_dir / "train.csv"),
                "validation_sha256": sha(data_dir / "validation.csv"),
                "validation_predictions": predictions.tolist()}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    report = {**evidence, "metrics": metrics, "result_fingerprint": fingerprint,
              "model_sha256": sha(out_dir / "model.joblib"),
              "python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: version(p) for p in CORE_PACKAGES}}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    pd.DataFrame({"actual": y_val, "predicted": predictions}).to_csv(out_dir / "predictions.csv", index=False)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.data_dir, args.config, args.out)
    print(json.dumps(result["metrics"]))
