#!/usr/bin/env python
"""Download the SMS Spam Collection into data/SMSSpamCollection and check it.

    python scripts/download_data.py                 # UCI first, GitHub mirror as fallback
    python scripts/download_data.py --from-file ~/Downloads/SMSSpamCollection
    python scripts/download_data.py --force         # re-download

Dataset: Almeida & Hidalgo, "SMS Spam Collection", UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/228/sms+spam+collection  (CC BY 4.0).

The file is validated by shape and class balance after loading (5572 rows, 4825 ham /
747 spam with pandas' default parser; see spamdetector/data.py). The sha256 is printed and
stored in data/MANIFEST.json so you can compare runs and machines.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spamdetector import config  # noqa: E402
from spamdetector.data import load_dataset  # noqa: E402

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Third-party mirror of the same file (tab separated, no header), used only if UCI is unreachable.
MIRROR_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "whatsapp-spam-detector/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https URLs)
        return resp.read()


def from_uci() -> bytes:
    with zipfile.ZipFile(io.BytesIO(fetch(UCI_URL))) as zf:
        return zf.read("SMSSpamCollection")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file", type=Path, help="use a local copy instead of downloading")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    ap.add_argument("--out", type=Path, default=config.data_path())
    args = ap.parse_args()

    out: Path = args.out
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to replace it). Validating...")
        load_dataset(out)
        print("OK")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.from_file:
        content, source = args.from_file.read_bytes(), str(args.from_file)
    else:
        try:
            print(f"Downloading from UCI: {UCI_URL}")
            content, source = from_uci(), UCI_URL
        except Exception as exc:  # network errors, bad zip, ...
            print(f"  UCI download failed ({exc.__class__.__name__}: {exc}). Trying the GitHub mirror.")
            try:
                content, source = fetch(MIRROR_URL), MIRROR_URL
            except Exception as exc2:
                print(f"  Mirror failed too ({exc2}).\n"
                      "Download the dataset manually from the UCI page above, unzip it and run:\n"
                      "  python scripts/download_data.py --from-file path/to/SMSSpamCollection")
                return 1

    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(content)
    try:
        df = load_dataset(tmp)  # raises if the file is not the expected dataset
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    shutil.move(tmp, out)

    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "file": out.name, "source": source, "sha256": digest, "rows_loaded": len(df),
        "ham": int((df.label == "ham").sum()), "spam": int((df.label == "spam").sum()),
        "license": "CC BY 4.0 (Almeida & Hidalgo, UCI ML Repository)",
    }
    (out.parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {out}\n  rows={manifest['rows_loaded']} ham={manifest['ham']} spam={manifest['spam']}\n  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
