"""Confidence calibration metrics.

A confidence number is only useful if it means something. These measures check
that: ECE, Brier score, and reliability bins. If a reranker exposes a confidence
value, the evaluation script reports these alongside it.

Nothing here assumes the confidence head is good. Poor calibration is reported as
poor calibration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Expected calibration error over equal-width confidence bins.

    Parameters
    ----------
    confidences: predicted confidences in [0, 1].
    correct: whether the associated prediction was correct.
    n_bins: number of reliability bins.

    Returns ``ece`` (the weighted mean absolute gap) plus the per-bin breakdown.
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"confidences ({len(confidences)}) and correct ({len(correct)}) "
            "must have the same length"
        )
    if not confidences:
        raise ValueError("cannot compute ECE on an empty sample")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    confidence = np.asarray(confidences, dtype=float)
    outcome = np.asarray(correct, dtype=float)

    if confidence.min() < 0.0 or confidence.max() > 1.0:
        raise ValueError("confidences must lie in [0, 1]")
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise ValueError("correct must contain only booleans")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: List[Dict[str, float]] = []

    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        # Last bin is closed on the right so confidence == 1.0 is included.
        mask = (
            (confidence > low) & (confidence <= high)
            if index == n_bins - 1
            else (confidence >= low) & (confidence < high)
        )
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_lower": float(low),
                    "bin_upper": float(high),
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        mean_confidence = float(confidence[mask].mean())
        accuracy = float(outcome[mask].mean())
        gap = abs(accuracy - mean_confidence)
        ece += (count / len(confidence)) * gap

        bins.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )

    return {
        "ece": float(ece),
        "n_bins": int(n_bins),
        "num_samples": int(len(confidence)),
        "bins": bins,
    }


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error between stated confidence and observed accuracy.

    Lower is better; 0 means the stated confidence always matched reality.
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"confidences ({len(confidences)}) and correct ({len(correct)}) "
            "must have the same length"
        )
    if not confidences:
        raise ValueError("cannot compute Brier score on an empty sample")

    confidence = np.asarray(confidences, dtype=float)
    outcome = np.asarray(correct, dtype=float)
    return float(np.mean((confidence - outcome) ** 2))


def negative_log_likelihood(
    confidences: Sequence[float], correct: Sequence[bool]
) -> float:
    """Log loss of binary outcomes under the stated confidences."""
    if len(confidences) != len(correct):
        raise ValueError(
            "confidences and correct must have the same length"
        )
    if not confidences:
        raise ValueError("cannot compute NLL on an empty sample")

    confidence = np.clip(np.asarray(confidences, dtype=float), 1e-7, 1 - 1e-7)
    outcome = np.asarray(correct, dtype=float)
    return float(-np.mean(outcome * np.log(confidence) + (1 - outcome) * np.log(1 - confidence)))


def calibration_curve(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> List[Tuple[float, float, int]]:
    """``(mean_confidence, accuracy, count)`` per non-empty bin."""
    result = expected_calibration_error(confidences, correct, n_bins=n_bins)
    return [
        (b["mean_confidence"], b["accuracy"], b["count"])
        for b in result["bins"]
        if b["count"] > 0
    ]


def abstention_report(
    scores: Sequence[float],
    confidences: Sequence[float],
    relevant: Sequence[bool],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> List[Dict[str, float]]:
    """Sweep abstention thresholds and report coverage versus accuracy.

    Lowering the threshold means answering more often. The interesting question
    is whether accuracy on the answered subset rises faster than coverage falls —
    if it does not, the confidence signal is not useful and the configuration
    should not enable abstention.
    """
    if not (len(scores) == len(confidences) == len(relevant)):
        raise ValueError("scores, confidences and relevant must have equal length")
    if not scores:
        raise ValueError("cannot build an abstention report on an empty sample")

    confidence = np.asarray(confidences, dtype=float)
    outcome = np.asarray(relevant, dtype=float)
    rows: List[Dict[str, float]] = []

    for threshold in thresholds:
        answered = confidence >= threshold
        count = int(answered.sum())
        if count == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "selective_accuracy": 0.0,
                    "num_answered": 0,
                    "num_abstained": int(len(confidence)),
                }
            )
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(count / len(confidence)),
                "selective_accuracy": float(outcome[answered].mean()),
                "num_answered": count,
                "num_abstained": int(len(confidence) - count),
            }
        )
    return rows


def evaluate_confidence(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 10,
) -> Dict[str, object]:
    """Bundle every calibration measure into one payload."""
    result = expected_calibration_error(confidences, outcomes, n_bins=n_bins)
    result["brier"] = brier_score(confidences, outcomes)
    result["nll"] = negative_log_likelihood(confidences, outcomes)
    result["mean_confidence"] = float(np.mean(confidences))
    result["accuracy"] = float(np.mean(np.asarray(outcomes, dtype=float)))
    return result


@torch.no_grad()
def confidence_from_tensor(
    logits: torch.Tensor, method: str = "sigmoid"
) -> torch.Tensor:
    """Turn raw confidence logits into probabilities."""
    if method == "sigmoid":
        return torch.sigmoid(logits)
    if method == "softmax":
        return torch.softmax(logits.reshape(-1, 2), dim=-1)[:, 1]
    raise ValueError(f"unsupported confidence method {method!r}")


def top_margin_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """top1 - top2 margin, or 0 when only one candidate was scored."""
    if scores.dim() != 2:
        raise ValueError(f"expected (B, N) scores, got {tuple(scores.shape)}")
    if scores.shape[1] == 1:
        return torch.zeros(scores.shape[0])
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    return sorted_scores[:, 0] - sorted_scores[:, 1]
