from pathlib import Path
import pandas as pd

def merge_submission_files():
    root = Path("data/processed")

    final_path = root / "final_submission_intervals_v5.csv"
    subm_path  = root / "submission_intervals_v5.csv"
    out_path   = root / "final_submission_intervals_merged_v5.csv"

    print("Loading submission files...")
    df_final = pd.read_csv(final_path)
    df_subm  = pd.read_csv(subm_path)

    print(f"  Final file:      {len(df_final)} rows, {len(df_final.columns)} cols")
    print(f"  Submission file: {len(df_subm)} rows, {len(df_subm.columns)} cols")

    # Only keep columns also present in df_final
    final_cols = list(df_final.columns)
    dropped_cols = [c for c in df_subm.columns if c not in final_cols]

    if dropped_cols:
        print("  Dropping extra columns from submission file:")
        for c in dropped_cols:
            print("    -", c)

    df_subm_trimmed = df_subm[final_cols]

    # Merge by vertical stacking
    merged = pd.concat([df_final, df_subm_trimmed], ignore_index=True)

    print(f"  Merged file:     {len(merged)} rows, {len(merged.columns)} cols")
    merged.to_csv(out_path, index=False)
    print(f"Saved merged file → {out_path}\n")


def truncate_features_file():
    root = Path("data/processed")

    src = root / "features_intervals_v5.csv"
    ref = root / "final_submission_intervals_v5.csv"
    out = root / "features_intervals_v5_truncated.csv"

    print("Truncating features_intervals_v5.csv ...")

    # Load reference and target
    df_ref = pd.read_csv(ref)
    df = pd.read_csv(src)

    print(f"  Original FEATURES: {len(df)} rows, {len(df.columns)} cols")

    # Keep only columns also present in final_submission_intervals_v5.csv
    keep_cols = [c for c in df.columns if c in df_ref.columns]
    drop_cols = [c for c in df.columns if c not in df_ref.columns]

    if drop_cols:
        print("  Dropping columns not in reference file:")
        for c in drop_cols:
            print("    -", c)

    df_trunc = df[keep_cols]

    print(f"  Truncated FEATURES: {len(df_trunc)} rows, {len(df_trunc.columns)} cols")
    df_trunc.to_csv(out, index=False)
    print(f"Saved truncated features file → {out}\n")


def truncate_submission_file():
    root = Path("data/processed")

    src = root / "submission_intervals_v5.csv"
    ref = root / "final_submission_intervals_v5.csv"
    out = root / "submission_intervals_v5_truncated.csv"

    print("Truncating submission_intervals_v5.csv ...")

    # Load reference and target
    df_ref = pd.read_csv(ref)
    df = pd.read_csv(src)

    print(f"  Original SUBMISSION: {len(df)} rows, {len(df.columns)} cols")

    # Keep only columns also present in final_submission_intervals_v5.csv
    keep_cols = [c for c in df.columns if c in df_ref.columns]
    drop_cols = [c for c in df.columns if c not in df_ref.columns]

    if drop_cols:
        print("  Dropping columns not in reference file:")
        for c in drop_cols:
            print("    -", c)

    df_trunc = df[keep_cols]

    print(f"  Truncated SUBMISSION: {len(df_trunc)} rows, {len(df_trunc.columns)} cols")
    df_trunc.to_csv(out, index=False)
    print(f"Saved truncated submission file → {out}\n")


def main():
    merge_submission_files()
    truncate_features_file()
    truncate_submission_file()
    print("Done.")


if __name__ == "__main__":
    main()
