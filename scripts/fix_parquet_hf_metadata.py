#!/usr/bin/env python3
"""Strip the embedded `huggingface` schema metadata from parquet files.

Some parquet files carry a `huggingface` key in their Arrow schema metadata
that records `datasets.Features` using a different (incompatible) `datasets`
library version. When such files are loaded with `datasets.load_dataset`
on a mismatched `datasets` version, `Features.from_arrow_schema` crashes with:

    TypeError: must be called with a dataclass type or instance

Stripping the `huggingface` metadata makes `datasets` fall back to inferring
features directly from the Arrow physical types, which avoids the crash.

Usage:
    python3 scripts/fix_parquet_hf_metadata.py \
        --src-dir /xpfs/fp4/dpj/data/gsm8k \
        --dst-dir /xpfs/fp4/gg/data/gsm8k
"""

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def strip_hf_metadata(src_path: Path, dst_path: Path) -> None:
    table = pq.read_table(src_path)
    schema = table.schema
    metadata = dict(schema.metadata or {})
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata.pop(b"huggingface", None) is not None:
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, dst_path)
        print(f"[stripped] {src_path} -> {dst_path}")
    else:
        shutil.copy2(src_path, dst_path)
        print(f"[copied, no huggingface metadata found] {src_path} -> {dst_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="Directory containing the original parquet files.")
    parser.add_argument("--dst-dir", type=Path, required=True, help="Directory to write the cleaned parquet files to.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train.parquet", "test.parquet"],
        help="Parquet filenames (relative to --src-dir) to process.",
    )
    args = parser.parse_args()

    for name in args.files:
        src_path = args.src_dir / name
        dst_path = args.dst_dir / name
        if not src_path.exists():
            print(f"[skip] {src_path} does not exist")
            continue
        strip_hf_metadata(src_path, dst_path)


if __name__ == "__main__":
    main()
