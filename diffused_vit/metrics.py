from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score


@dataclass
class BinaryMetrics:
    auc: float
    accuracy: float
    precision: float
    sensitivity: float
    specificity: float

    def to_dict(self):
        return asdict(self)


def calculate_binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> BinaryMetrics:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    return BinaryMetrics(
        auc=auc,
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
    )
