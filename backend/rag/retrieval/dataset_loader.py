from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORY_TO_DOMAIN = {
    "academic_rules": "HR_ADMIN",
    "fees": "FEES",
    "fee": "FEES",
    "it": "IT",
    "wifi": "IT",
    "email": "IT",
    "technology": "IT",
    "hostel": "FACILITIES",
    "transport": "FACILITIES",
    "maintenance": "FACILITIES",
    "campus_facility": "FACILITIES",
    "facility": "FACILITIES",
}


def load_and_normalize_chunks(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load Member 2's JSONL chunks and normalize them
    to the retrieval contract used by Member 3.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}"
        )

    documents: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}."
                ) from exc

            chunk_id = record.get("chunk_id")
            text = record.get("text")
            metadata = record.get("metadata")

            if not chunk_id:
                raise ValueError(
                    f"Missing chunk_id on line {line_number}."
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing text on line {line_number}."
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Metadata must be an object "
                    f"on line {line_number}."
                )

            category = str(
                metadata.get("category", "")
            ).strip().lower()

            domain = CATEGORY_TO_DOMAIN.get(
                category
            )

            if domain is None:
                raise ValueError(
                    f"Unsupported category '{category}' "
                    f"on line {line_number}."
                )

            pages = metadata.get(
                "pages",
                [],
            )

            if not isinstance(pages, list):
                pages = [pages]

            clean_pages = [
                int(page)
                for page in pages
                if page is not None
            ]

            documents.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": text.strip(),
                    "domain": domain,
                    "source_title": str(
                        metadata.get(
                            "document",
                            "Unknown document",
                        )
                    ),
                    "source_url": metadata.get(
                        "source_url"
                    ),
                    "page": (
                        clean_pages[0]
                        if clean_pages
                        else None
                    ),
                    "pages": clean_pages,
                    "category": category,
                    "source": metadata.get(
                        "source"
                    ),
                }
            )

    if not documents:
        raise ValueError(
            "No valid chunks were found."
        )

    return documents


__all__ = [
    "CATEGORY_TO_DOMAIN",
    "load_and_normalize_chunks",
]