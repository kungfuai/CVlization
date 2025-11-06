import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_svmlight_file


BASE_DATA_CACHE = Path(
    os.environ.get("CVL_DATA_CACHE", Path.home() / ".cache" / "cvlization" / "data")
)
DATASET_NAME = "lightgbm_lambdarank_demo"
DATASET_DIR = BASE_DATA_CACHE / DATASET_NAME

DATA_FILES = [
    "rank.train",
    "rank.train.query",
    "rank.test",
    "rank.test.query",
]

RAW_BASE_URL = "https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/lambdarank/"

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "ranking_lightgbm.txt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
SAMPLE_CANDIDATES_PATH = ARTIFACTS_DIR / "sample_candidates.csv"
SAMPLE_RANKINGS_PATH = ARTIFACTS_DIR / "sample_rankings.csv"


def ensure_dataset_downloaded() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for filename in DATA_FILES:
        destination = DATASET_DIR / filename
        if destination.exists():
            print(f"Using cached dataset file: {destination}")
            continue
        url = RAW_BASE_URL + filename
        print(f"Downloading {url} -> {destination}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)


def load_group_file(path: Path) -> List[int]:
    groups: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(int(line))
    if not groups:
        raise ValueError(f"No groups found in {path}")
    return groups


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, List[int]]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split '{split}'")

    prefix = "rank.train" if split == "train" else "rank.test"
    data_path = DATASET_DIR / prefix
    group_path = DATASET_DIR / f"{prefix}.query"

    groups = load_group_file(group_path)
    features, labels = load_svmlight_file(str(data_path), zero_based=True)
    labels = labels.astype(np.float32)

    return features, labels, groups


def mean_ndcg(labels: np.ndarray, scores: np.ndarray, groups: Iterable[int], k: int) -> float:
    from sklearn.metrics import ndcg_score

    results: List[float] = []
    start = 0
    for group_size in groups:
        end = start + group_size
        truth = labels[start:end]
        pred = scores[start:end]
        current_k = min(k, group_size)
        if current_k == 0:
            results.append(0.0)
        else:
            results.append(
                float(ndcg_score([truth], [pred], k=current_k, ignore_ties=False))
            )
        start = end
    return float(np.mean(results))


def average_precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1]
    if k is not None:
        order = order[:k]
    sorted_labels = labels[order]

    relevant = (sorted_labels > 0).astype(float)
    total_relevant = float((labels > 0).sum())
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    hits = 0.0
    for idx, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1.0
            precision_sum += hits / idx

    return float(precision_sum / total_relevant)


def mean_average_precision(labels: np.ndarray, scores: np.ndarray, groups: Iterable[int], k: int) -> float:
    results: List[float] = []
    start = 0
    for group_size in groups:
        end = start + group_size
        results.append(average_precision_at_k(labels[start:end], scores[start:end], k))
        start = end
    return float(np.mean(results))


def create_sample_artifacts(
    features,
    labels: np.ndarray,
    groups: List[int],
    scores: np.ndarray,
    feature_names: List[str],
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    query_ids = np.repeat(np.arange(len(groups)), groups)
    sample_query_ids = np.unique(query_ids)[:2]
    sample_mask = np.isin(query_ids, sample_query_ids)
    sample_indices = np.nonzero(sample_mask)[0]

    if hasattr(features, "toarray"):
        sample_features = features[sample_indices].toarray()
    else:
        sample_features = np.asarray(features[sample_indices])

    candidate_df = pd.DataFrame(sample_features, columns=feature_names)
    candidate_df.insert(0, "query_id", query_ids[sample_indices] + 1)
    candidate_df.to_csv(SAMPLE_CANDIDATES_PATH, index=False)
    print(f"Sample candidates saved to {SAMPLE_CANDIDATES_PATH}")

    sample_scores = scores[sample_indices]
    ranking_df = candidate_df.copy()
    ranking_df["relevance"] = labels[sample_indices]
    ranking_df["score"] = sample_scores
    ranking_df = ranking_df.sort_values(["query_id", "score"], ascending=[True, False])
    ranking_df["rank"] = ranking_df.groupby("query_id").cumcount() + 1
    ranking_df.to_csv(SAMPLE_RANKINGS_PATH, index=False)
    print(f"Sample rankings saved to {SAMPLE_RANKINGS_PATH}")


def train_model(train_set: lgb.Dataset, valid_set: lgb.Dataset, params: Dict[str, object]) -> lgb.Booster:
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(50),
    ]
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(os.environ.get("NUM_BOOST_ROUND", 400)),
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return booster


def main() -> None:
    ensure_dataset_downloaded()

    train_features, train_labels, train_groups = load_split("train")
    test_features, test_labels, test_groups = load_split("test")

    feature_names = [f"f{i+1}" for i in range(train_features.shape[1])]

    train_groups_arr = np.array(train_groups, dtype=int)
    if train_groups_arr.size < 2:
        raise ValueError("Expected at least two query groups in the training split.")
    valid_fraction = float(os.environ.get("VALID_GROUP_FRACTION", 0.2))
    valid_group_count = max(1, int(len(train_groups_arr) * valid_fraction))
    if valid_group_count >= len(train_groups_arr):
        valid_group_count = len(train_groups_arr) - 1
    if valid_group_count <= 0:
        valid_group_count = 1
    train_group_count = len(train_groups_arr) - valid_group_count
    if train_group_count <= 0:
        train_group_count = len(train_groups_arr) - 1
        valid_group_count = 1

    train_group_sizes = train_groups_arr[:train_group_count].tolist()
    valid_group_sizes = train_groups_arr[train_group_count:].tolist()

    train_row_cut = int(np.sum(train_group_sizes))
    valid_row_cut = int(np.sum(valid_group_sizes))

    print(
        f"Split {len(train_groups_arr)} queries into "
        f"{train_group_count} train / {valid_group_count} validation queries "
        f"({train_row_cut} + {valid_row_cut} documents)."
    )

    train_dataset = lgb.Dataset(
        train_features[:train_row_cut],
        label=train_labels[:train_row_cut],
        group=train_group_sizes,
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        train_features[train_row_cut : train_row_cut + valid_row_cut],
        label=train_labels[train_row_cut : train_row_cut + valid_row_cut],
        group=valid_group_sizes,
        free_raw_data=False,
    )

    train_dataset.set_feature_name(feature_names)
    valid_dataset.set_feature_name(feature_names)

    params = {
        "objective": "lambdarank",
        "metric": ["ndcg", "map"],
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.1,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "deterministic": True,
        "boosting_type": os.environ.get("BOOSTING_TYPE", "gbdt"),
    }

    model = train_model(train_dataset, valid_dataset, params)

    test_scores = model.predict(test_features)

    metrics = {
        "ndcg@5": mean_ndcg(test_labels, test_scores, test_groups, k=5),
        "ndcg@10": mean_ndcg(test_labels, test_scores, test_groups, k=10),
        "map@10": mean_average_precision(test_labels, test_scores, test_groups, k=10),
        "best_iteration": int(model.best_iteration or 0),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload: Dict[str, float | int] = {k: float(v) for k, v in metrics.items() if k != "best_iteration"}
    metrics_payload["best_iteration"] = metrics["best_iteration"]
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}: {metrics}")

    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    with FEATURE_NAMES_PATH.open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")

    create_sample_artifacts(
        test_features,
        test_labels,
        test_groups,
        test_scores,
        feature_names,
    )


if __name__ == "__main__":
    main()
