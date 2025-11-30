#!/usr/bin/env python3
# 11_interval_data_from_parquet.py
import argparse
import sys
from pathlib import Path
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(description="Print a row by index from a Parquet file.")
    parser.add_argument("--idx", type=int, required=True, help="0-based row index to display")
    parser.add_argument("--file", default="prc-2025-datasets/fuel_train.parquet",
                        help="Path to the Parquet file (default: prc-2025-datasets/fuel_train.parquet)")
    args = parser.parse_args()

    parquet_path = Path(args.file)
    if not parquet_path.exists():
        print(f"ERROR: File not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    if args.idx < 0:
        print("ERROR: --idx must be >= 0", file=sys.stderr)
        sys.exit(1)

    pf = pq.ParquetFile(parquet_path)

    # Stream through batches so we don't have to load the full file
    target = args.idx
    seen = 0
    for batch in pf.iter_batches():
        batch_len = batch.num_rows
        if target < seen + batch_len:
            # The row is inside this batch
            offset = target - seen
            # Convert the single row slice to a dict of {column: value}
            row_dict = {name: batch.column(i)[offset].as_py()
                        for i, name in enumerate(batch.schema.names)}
            headers = list(row_dict.keys())
            values = [row_dict[h] for h in headers]

            # Print header line then value line (tab-separated)
            print("\t".join(headers))
            print("\t".join("" if v is None else str(v) for v in values))
            return
        seen += batch_len

    # If we got here, index was out of range
    total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))
    print(f"ERROR: --idx {args.idx} is out of range (total rows: {total_rows})", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
