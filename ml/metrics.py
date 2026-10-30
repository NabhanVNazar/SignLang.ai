"""Gloss-sequence metrics based on token and character edit distance."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return Levenshtein edit distance for a sequence of tokens."""
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def _aligned_token_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    rows, columns = len(reference), len(hypothesis)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        costs[row][0] = row
    for column in range(columns + 1):
        costs[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1]
                + (reference[row - 1] != hypothesis[column - 1]),
            )

    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()
    row, column = rows, columns
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == hypothesis[column - 1]
            and costs[row][column] == costs[row - 1][column - 1]
        ):
            true_positive[reference[row - 1]] += 1
            row -= 1
            column -= 1
        elif (
            row
            and column
            and costs[row][column] == costs[row - 1][column - 1] + 1
        ):
            false_negative[reference[row - 1]] += 1
            false_positive[hypothesis[column - 1]] += 1
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            false_negative[reference[row - 1]] += 1
            row -= 1
        else:
            false_positive[hypothesis[column - 1]] += 1
            column -= 1
    return true_positive, false_positive, false_negative


def sequence_metrics(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    minimum_per_gloss_occurrences: int = 5,
) -> dict[str, object]:
    if not references or len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same non-zero length")
    if minimum_per_gloss_occurrences < 1:
        raise ValueError("minimum_per_gloss_occurrences must be positive")
    if any(not sequence for sequence in references):
        raise ValueError("reference gloss sequences must not be empty")

    total_token_edits = sum(
        edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    )
    reference_token_count = sum(len(sequence) for sequence in references)
    exact_matches = sum(
        list(reference) == list(hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    )
    reference_text = [" ".join(sequence) for sequence in references]
    hypothesis_text = [" ".join(sequence) for sequence in hypotheses]
    character_edits = sum(
        edit_distance(list(reference), list(hypothesis))
        for reference, hypothesis in zip(reference_text, hypothesis_text)
    )
    character_count = sum(len(text) for text in reference_text)

    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    for reference, hypothesis in zip(references, hypotheses):
        reference_counts.update(reference)
        aligned = _aligned_token_counts(reference, hypothesis)
        true_positive.update(aligned[0])
        false_positive.update(aligned[1])
        false_negative.update(aligned[2])

    per_gloss: dict[str, dict[str, float | int]] = {}
    for gloss, count in sorted(reference_counts.items()):
        if count < minimum_per_gloss_occurrences:
            continue
        tp, fp, fn = (
            true_positive[gloss],
            false_positive[gloss],
            false_negative[gloss],
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_gloss[gloss] = {
            "reference_occurrences": count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "gloss_token_error_rate": total_token_edits / reference_token_count,
        "character_error_rate": character_edits / character_count
        if character_count
        else 0.0,
        "exact_match_accuracy": exact_matches / len(references),
        "sample_count": len(references),
        "reference_gloss_token_count": reference_token_count,
        "per_gloss": per_gloss,
        "per_gloss_minimum_reference_occurrences": minimum_per_gloss_occurrences,
    }
