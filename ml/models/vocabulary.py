"""Deterministic gloss vocabulary for CTC sequence recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


class GlossVocabulary:
    """Maps gloss tokens to IDs, reserving zero for the CTC blank."""

    FORMAT_VERSION = 1
    BLANK_ID = 0

    def __init__(self, token_to_id: dict[str, int]) -> None:
        if not token_to_id:
            raise ValueError("vocabulary must contain at least one gloss")
        if any(not isinstance(token, str) or not token for token in token_to_id):
            raise ValueError("gloss tokens must be non-empty strings")
        expected = set(range(1, len(token_to_id) + 1))
        if set(token_to_id.values()) != expected:
            raise ValueError("gloss IDs must be unique consecutive integers starting at 1")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {index: token for token, index in token_to_id.items()}

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[str]]) -> GlossVocabulary:
        tokens = sorted({token for sequence in sequences for token in sequence})
        if not tokens:
            raise ValueError("training sequences contain no glosses")
        return cls({token: index for index, token in enumerate(tokens, start=1)})

    @property
    def class_count(self) -> int:
        return len(self.token_to_id) + 1

    def encode(self, sequence: Sequence[str]) -> list[int]:
        unknown = sorted(set(sequence) - self.token_to_id.keys())
        if unknown:
            raise ValueError(
                "glosses are absent from the training vocabulary: "
                + ", ".join(unknown)
            )
        return [self.token_to_id[token] for token in sequence]

    def decode_ctc(self, frame_ids: Sequence[int]) -> list[str]:
        decoded: list[str] = []
        previous_id: int | None = None
        for class_id in frame_ids:
            if class_id < 0 or class_id >= self.class_count:
                raise ValueError(f"class ID {class_id} is outside the vocabulary")
            if class_id != self.BLANK_ID and class_id != previous_id:
                decoded.append(self.id_to_token[class_id])
            previous_id = class_id
        return decoded

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "blank_id": self.BLANK_ID,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> GlossVocabulary:
        if not isinstance(value, dict):
            raise ValueError("vocabulary file must contain a JSON object")
        format_version = value.get("format_version")
        if isinstance(format_version, bool) or format_version != cls.FORMAT_VERSION:
            raise ValueError("unsupported vocabulary format version")
        blank_id = value.get("blank_id")
        if isinstance(blank_id, bool) or blank_id != cls.BLANK_ID:
            raise ValueError("vocabulary blank_id must be zero")
        token_to_id = value.get("token_to_id")
        if not isinstance(token_to_id, dict) or any(
            not isinstance(token, str)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            for token, class_id in token_to_id.items()
        ):
            raise ValueError("vocabulary token_to_id must map strings to integer IDs")
        return cls(token_to_id)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> GlossVocabulary:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
