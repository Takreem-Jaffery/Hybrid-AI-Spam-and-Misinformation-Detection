#!/usr/bin/env python
"""Download the licensed article-veracity and clickbait training/evaluation data.

    python scripts/download_article_datasets.py

Only article XML/metadata and Clickbait JSONL files are extracted. Images, multi-gigabyte
web archives, and the Clickbait-17 hidden test set are intentionally excluded.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spamdetector import config  # noqa: E402


DATASETS = {
    "buzzfeed_articles": {
        "url": "https://zenodo.org/api/records/1239675/files/articles.zip/content",
        "filename": "buzzfeed-webis-articles.zip",
        "md5": "5899194f95016cc90aae5e6d0e7e5042",
    },
    "buzzfeed_metadata": {
        "url": "https://zenodo.org/api/records/1239675/files/overview.csv/content",
        "filename": "buzzfeed-webis-overview.csv",
        "md5": "eab4909641bd707fffe514e9cd201430",
    },
    "clickbait_train": {
        "url": "https://zenodo.org/api/records/5530410/files/clickbait17-train-170331.zip/content",
        "filename": "webis-clickbait-17-train-170331.zip",
        "md5": "738075eaf9c4d3af9c16808f5d16ce00",
    },
    "clickbait_validation": {
        "url": "https://zenodo.org/api/records/5530410/files/clickbait17-train-170630.zip/content",
        "filename": "webis-clickbait-17-validation-170630.zip",
        "md5": "8f074f3f035393b81a559bfc8bd9a467",
    },
}


def download(url: str, destination: Path, expected_md5: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _md5(destination) == expected_md5:
        print(f"Already verified: {destination.name}")
        return expected_md5

    previous_name = destination.with_name("webis-clickbait-17-train.zip")
    if not destination.exists() and previous_name.exists() and _md5(previous_name) == expected_md5:
        previous_name.replace(destination)
        print(f"Reused verified archive as: {destination.name}")
        return expected_md5

    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()
    request = urllib.request.Request(url, headers={"User-Agent": "MessageRiskAnalyzer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", "0"))
            next_report = 100 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                if stream.tell() >= next_report:
                    size = stream.tell() / (1024 * 1024)
                    suffix = f" / {total / (1024 * 1024):.0f} MiB" if total else ""
                    print(f"Downloading {destination.name}: {size:.0f} MiB{suffix}", flush=True)
                    next_report += 100 * 1024 * 1024
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    actual = digest.hexdigest()
    if actual != expected_md5:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {destination.name}: expected {expected_md5}, got {actual}")
    partial.replace(destination)
    print(f"Downloaded and verified: {destination.name}")
    return actual


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_zip(
    archive: Path,
    destination: Path,
    *,
    include_prefix: str | None = None,
    include_suffixes: set[str] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        root = destination.resolve()
        for item in source.infolist():
            archive_name = item.filename.replace("\\", "/")
            if include_prefix and not archive_name.lower().startswith(include_prefix.lower()):
                continue
            relative_name = archive_name[len(include_prefix):] if include_prefix else archive_name
            path = Path(relative_name)
            if include_suffixes and not item.is_dir() and path.suffix.lower() not in include_suffixes:
                continue
            if item.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in dataset archive: {item.filename}")
            target = (destination / path).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in dataset archive: {item.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)


def main() -> None:
    root = config.article_data_dir()
    raw = root / "raw"
    checksums = {}
    paths = {}
    for name, metadata in DATASETS.items():
        path = raw / metadata["filename"]
        checksums[name] = download(metadata["url"], path, metadata["md5"])
        paths[name] = path

    extract_zip(paths["buzzfeed_articles"], root / "buzzfeed_webis" / "articles", include_prefix="articles/")
    (root / "buzzfeed_webis" / "overview.csv").write_bytes(paths["buzzfeed_metadata"].read_bytes())
    extract_zip(paths["clickbait_train"], root / "clickbait17" / "train", include_suffixes={".jsonl"})
    extract_zip(
        paths["clickbait_validation"],
        root / "clickbait17" / "validation",
        include_suffixes={".jsonl"},
    )
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "buzzfeed_webis": {
                        "citation": "Potthast et al., BuzzFeed-Webis Fake News Corpus 2016, https://doi.org/10.5281/zenodo.1239675",
                        "license": "CC BY 4.0",
                        "source": "https://doi.org/10.5281/zenodo.1239675",
                        "downloaded_files": {
                            DATASETS["buzzfeed_articles"]["filename"]: checksums["buzzfeed_articles"],
                            DATASETS["buzzfeed_metadata"]["filename"]: checksums["buzzfeed_metadata"],
                        },
                        "excluded": ["web-archives.zip"],
                    },
                    "webis_clickbait17": {
                        "citation": "Potthast et al., Webis Clickbait Corpus 2017, https://doi.org/10.5281/zenodo.5530410",
                        "license": "CC BY 4.0",
                        "source": "https://doi.org/10.5281/zenodo.5530410",
                        "downloaded_files": {
                            DATASETS["clickbait_train"]["filename"]: checksums["clickbait_train"],
                            DATASETS["clickbait_validation"]["filename"]: checksums["clickbait_validation"],
                        },
                        "excluded": ["media files", "clickbait17-test-170720.zip", "original WARC archives"],
                    },
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Datasets are ready in {root}")


if __name__ == "__main__":
    main()
