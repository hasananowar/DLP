#!/usr/bin/env python3
"""
extra_preprocess.py

A simple script to load, process, and save CSV data:
- Reads an input CSV file specified by the user
- Sorts rows by the `time` column (ascending)
- Drops the optional `idx` and `ext_roll` columns if present
- Writes the cleaned DataFrame to an output CSV file

Usage:
    python extra_preprocess.py <input_csv> <output_csv>
"""
import argparse
import os
import pandas as pd


def process_csv(input_path: str, output_path: str) -> None:
    """
    Load a CSV file, sort by the 'time' column, drop unwanted columns, and save to a new CSV.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path where the processed CSV will be saved.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if 'time' not in df.columns:
        raise KeyError("Column 'time' is required in the input CSV.")

    # Sort rows by time ascending
    df = df.sort_values(by='time', ascending=True, ignore_index=True)

    # Drop optional columns if they exist
    for col in ('idx', 'ext_roll'):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save processed DataFrame
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sort a CSV by time and drop unwanted columns."
    )
    parser.add_argument(
        'input_csv',
        help='Path to the input CSV file'
    )
    parser.add_argument(
        'output_csv',
        help='Path to save the processed CSV file'
    )
    args = parser.parse_args()
    process_csv(args.input_csv, args.output_csv)


if __name__ == '__main__':
    main()
