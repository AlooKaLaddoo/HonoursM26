#!/usr/bin/env python3
"""Build a CSV of subject-session labels sorted by age_acq_time.

This script scans a BIDS-style infant dataset for every ``*_scans.tsv`` file,
extracts the ``age_acq_time`` value, and writes a CSV ordered from youngest to
oldest.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> tuple[Path, Path]:
	"""Return the input dataset root and output CSV path."""
	root_dir = Path(argv[1]) if len(argv) > 1 else Path("Infants_data")
	output_file = Path(argv[2]) if len(argv) > 2 else Path ("age_acq_time_sorted.csv")
	return root_dir, output_file


def collect_age_rows(root_dir: Path) -> list[dict[str, object]]:
	"""Collect label and age rows from all scans.tsv files under root_dir."""
	rows: list[dict[str, object]] = []

	for scan_path in sorted(root_dir.rglob("*_scans.tsv")):
		if not scan_path.is_file():
			continue

		subject_id = scan_path.parent.parent.name
		session_id = scan_path.parent.name
		label = f"{subject_id}_{session_id}"

		with scan_path.open(newline="", encoding="utf-8") as handle:
			reader = csv.DictReader(handle, delimiter="\t")
			if not reader.fieldnames or "age_acq_time" not in reader.fieldnames:
				continue

			for row in reader:
				age_text = (row.get("age_acq_time") or "").strip()
				if not age_text:
					continue

				try:
					age_value = float(age_text)
				except ValueError:
					continue

				rows.append({"label": label, "age_acq_time": age_value})

	rows.sort(key=lambda item: (item["age_acq_time"], item["label"]))
	return rows


def write_csv(rows: list[dict[str, object]], output_file: Path) -> None:
	"""Write the sorted rows to output_file."""
	output_file.parent.mkdir(parents=True, exist_ok=True)

	with output_file.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=["label", "age_acq_time"])
		writer.writeheader()
		writer.writerows(rows)


def main(argv: list[str]) -> int:
	root_dir, output_file = parse_args(argv)

	if not root_dir.is_dir():
		print(f"Error: root directory not found: {root_dir}", file=sys.stderr)
		return 1

	rows = collect_age_rows(root_dir)
	write_csv(rows, output_file)

	print(f"Wrote {len(rows)} rows to {output_file}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv))
