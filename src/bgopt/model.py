"""Lightweight workload models with experiment-aware evaluation splits."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .features import MODEL_FEATURES

def _libs() -> tuple[Any, ...]:
    import joblib, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    return joblib, pd, HistGradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier, SimpleImputer, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, GroupShuffleSplit, train_test_split, Pipeline

def _classifier(kind: str, hist: Any, forest: Any, tree: Any, seed: int) -> Any:
    if kind == "hist_gradient_boosting": return hist(max_iter=120, learning_rate=.08, max_leaf_nodes=15, random_state=seed)
    if kind == "random_forest": return forest(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1)
    if kind == "decision_tree": return tree(max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=seed)
    raise ValueError("model_kind must be hist_gradient_boosting, random_forest, or decision_tree")

def _group_column(frame: Any, requested: str | None) -> str | None:
    if requested:
        if requested not in frame: raise ValueError(f"Requested group column is missing: {requested}")
        return requested
    return next((name for name in ("experiment_id", "group", "scenario") if name in frame and frame[name].nunique() >= 3), None)

def _metrics(y: Any, pred: Any, accuracy: Any, report: Any, matrix: Any, prf: Any) -> dict[str, Any]:
    labels = sorted(set(y).union(set(pred)))
    precision, recall, f1, _ = prf(y, pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy(y, pred)), "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1), "labels": labels, "confusion_matrix": matrix(y, pred, labels=labels).tolist(), "classification_report": report(y, pred, output_dict=True, zero_division=0)}

def train(dataset_path: str | Path, model_path: str | Path, report_path: str | Path | None = None, random_state: int = 42, model_kind: str = "hist_gradient_boosting", group_column: str | None = None) -> dict[str, Any]:
    joblib, pd, hist, forest, tree, imputer, accuracy, creport, matrix, prf, group_split, row_split, pipeline_cls = _libs()
    frame = pd.read_csv(dataset_path)
    missing = set(MODEL_FEATURES).difference(frame.columns)
    if missing or "workload_label" not in frame: raise ValueError(f"Dataset needs workload_label and all model features; missing: {sorted(missing)}")
    x, y = frame.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce"), frame["workload_label"].astype(str)
    counts = y.value_counts()
    if len(counts) < 2 or len(frame) < 12 or int(counts.min()) < 3: raise ValueError("Training needs 12 rows, two classes, and three rows per class")
    group_name = _group_column(frame, group_column)
    if group_name:
        groups = frame[group_name].astype(str)
        group_labels = frame.groupby(group_name)["workload_label"].nunique()
        if groups.nunique() < 3:
            raise ValueError("Group split needs three independent groups")
        if int(group_labels.max()) == 1:
            by_label = frame[[group_name, "workload_label"]].drop_duplicates().groupby("workload_label")[group_name].apply(lambda values: sorted(values.astype(str))).to_dict()
            if any(len(values) < 3 for values in by_label.values()):
                raise ValueError("Every workload class needs three independent groups")
            test_groups = {values[-1] for values in by_label.values()}
            validation_groups = {values[-2] for values in by_label.values()}
            test_mask, val_mask = groups.isin(test_groups), groups.isin(validation_groups)
            train_mask = ~(test_mask | val_mask)
            x_train, x_val, x_test = x.loc[train_mask], x.loc[val_mask], x.loc[test_mask]
            y_train, y_val, y_test = y.loc[train_mask], y.loc[val_mask], y.loc[test_mask]
            strategy = "grouped_per_class"
        else:
            splitter = group_split(test_size=.25, random_state=random_state)
            train_idx, test_idx = next(splitter.split(x, y, groups))
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            x_val, y_val = x_train.iloc[0:0], y_train.iloc[0:0]
            strategy = "grouped"
    else:
        x_train, x_test, y_train, y_test = row_split(x, y, test_size=.25, random_state=random_state)
        x_val, y_val, strategy = x_test.iloc[:0], y_test.iloc[:0], "row_stratified_fallback"
    pipeline = pipeline_cls([("imputer", imputer(strategy="median")), ("classifier", _classifier(model_kind, hist, forest, tree, random_state))])
    pipeline.fit(x_train, y_train)
    output: dict[str, Any] = {"feature_names": list(MODEL_FEATURES), "classes": sorted(y.unique().tolist()), "model_kind": model_kind, "split_strategy": strategy, "group_column": group_name, "rows": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)}, "class_counts": {str(k): int(v) for k, v in counts.items()}, "test": _metrics(y_test, pipeline.predict(x_test), accuracy, creport, matrix, prf)}
    if len(x_val): output["validation"] = _metrics(y_val, pipeline.predict(x_val), accuracy, creport, matrix, prf)
    if strategy != "grouped": output["leakage_warning"] = "No experiment/scenario/group split was possible. This row-level result is a development check, not a research claim."
    target = Path(model_path); target.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"pipeline": pipeline, "feature_names": list(MODEL_FEATURES), "classes": output["classes"], "report": output}, target)
    if report_path:
        report_target = Path(report_path); report_target.parent.mkdir(parents=True, exist_ok=True); report_target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output

def load(model_path: str | Path) -> dict[str, Any]:
    import joblib
    payload = joblib.load(model_path)
    if payload.get("feature_names") != list(MODEL_FEATURES): raise ValueError("Model feature schema does not match this BGOpt version")
    return payload

def predict(model_payload: dict[str, Any], record: dict[str, Any]) -> tuple[str, float, str]:
    import pandas as pd
    frame = pd.DataFrame([{name: record.get(name, 0.0) for name in MODEL_FEATURES}]); pipeline = model_payload["pipeline"]
    return str(pipeline.predict(frame)[0]), float(max(pipeline.predict_proba(frame)[0])) if hasattr(pipeline, "predict_proba") else 1.0, "model"

def evaluate(dataset_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    _, pd, _, _, _, _, accuracy, creport, matrix, prf, _, _, _ = _libs(); frame = pd.read_csv(dataset_path); payload = load(model_path)
    return {"rows": len(frame), "metrics_on_provided_dataset": _metrics(frame["workload_label"].astype(str), payload["pipeline"].predict(frame.loc[:, MODEL_FEATURES]), accuracy, creport, matrix, prf), "warning": "This score is unbiased only for an untouched held-out experiment/scenario capture."}
