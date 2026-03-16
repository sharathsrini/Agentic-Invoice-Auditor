"""Invoice Watcher Tool -- scans a directory for invoice+meta file pairs.

Finds actual attachment files by matching the stem prefix from .meta.json
filenames, ignoring the meta-declared extension. This handles cases like
INV_DE_004 where the meta declares .docx but the file is actually .pdf.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def invoice_watcher_tool(directory: str) -> list[dict]:
    """Scan directory for unprocessed invoice+meta file pairs.

    Finds actual attachment files by matching stem prefix, ignoring
    the meta-declared extension.

    Args:
        directory: Path to the incoming invoice directory.

    Returns:
        List of dicts with 'file_path' and 'meta' keys, sorted by filename.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_files = os.listdir(directory)
    seen = set()
    results = []

    for fname in sorted(all_files):
        if not fname.endswith(".meta.json"):
            continue

        stem = fname.replace(".meta.json", "")

        # Find actual attachment file on disk (ignore meta-declared extension)
        attachment = None
        for f in sorted(all_files):
            if f.startswith(stem) and not f.endswith(".meta.json"):
                attachment = f
                break

        if attachment and stem not in seen:
            seen.add(stem)
            meta_path = dir_path / fname
            with open(meta_path) as mf:
                meta = json.load(mf)
            results.append({
                "file_path": str(dir_path / attachment),
                "meta": meta,
            })
        elif not attachment:
            logger.warning(
                "Watcher: no attachment found for %s, skipping", fname
            )

    logger.info("Watcher found %d invoice(s) in %s", len(results), directory)
    return results
