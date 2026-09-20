from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

# ============================================================
# Count total trajectories for Chengdu / Xi'an / Beijing
#
# Purpose:
#   Run ONCE before index-scaling experiments.
#   The resulting totals are reused by both Ours and Baseline.
#
# This script does NOT build indexes and does NOT create subsets.
#
# Chengdu / Xi'an:
#   Read trajectory_globs from:
#       E:\VTRQ_baseline\configs\chengdu.json
#       E:\VTRQ_baseline\configs\xian.json
#
#   Count top-level JSON array elements with a streaming scanner.
#   It does NOT json.loads() the whole trajectory file.
#
# Beijing:
#   Count CSV data rows from:
#       E:\MMR_Trajectory_range\beijing_preprocessed\
#       beijing_trajectories.csv
#
# Output:
#   trajectory_totals.json
#   trajectory_totals.csv
# ============================================================

BASELINE_ROOT = Path(r"E:\VTRQ_baseline")
OURS_ROOT = Path(r"E:\MMR_Trajectory_range")

CONFIG_FILES = {
    "chengdu": BASELINE_ROOT / "configs" / "chengdu.json",
    "xian": BASELINE_ROOT / "configs" / "xian.json",
}

BEIJING_FILE = (
    OURS_ROOT
    / "beijing_preprocessed"
    / "beijing_trajectories.csv"
)

OUTPUT_JSON = Path(__file__).resolve().parent / "trajectory_totals.json"
OUTPUT_CSV = Path(__file__).resolve().parent / "trajectory_totals.csv"


def _expand_trajectory_files(
    config_file: Path,
) -> list[Path]:
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    cfg = json.loads(
        config_file.read_text(
            encoding="utf-8-sig"
        )
    )

    patterns = cfg.get(
        "trajectory_globs",
        [],
    )

    if not patterns:
        raise ValueError(
            f"{config_file}: missing trajectory_globs"
        )

    found: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        p = Path(str(pattern))

        if p.is_absolute():
            candidates = [str(p)]
        else:
            candidates = [
                str(config_file.parent / p),
                str(BASELINE_ROOT / p),
                str(p),
            ]

        for candidate in candidates:
            for value in sorted(
                glob.glob(candidate)
            ):
                path = Path(value).resolve()
                key = str(path).lower()

                if key not in seen:
                    seen.add(key)
                    found.append(path)

    found.sort(
        key=lambda p: str(p).lower()
    )

    if not found:
        raise FileNotFoundError(
            f"{config_file}: no trajectory files matched"
        )

    return found


def count_top_level_json_array(
    path: Path,
    chunk_size: int = 4 * 1024 * 1024,
) -> int:
    """
    Count elements of the FIRST top-level JSON array without
    constructing trajectory objects in memory.

    Existing Chengdu/Xi'an files may contain an extra unmatched
    closing ']' after a valid top-level JSON value. Therefore this
    scanner stops immediately when the first top-level array closes.

    Correctly ignores commas/brackets/braces inside JSON strings.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    square_depth = 0
    brace_depth = 0

    in_string = False
    escape = False

    top_started = False
    top_has_value = False
    separators = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="strict",
    ) as f:

        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            for ch in chunk:

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    # A string beginning directly inside the
                    # top-level array is itself a top-level value.
                    if (
                        top_started
                        and square_depth == 1
                        and brace_depth == 0
                    ):
                        top_has_value = True

                    in_string = True
                    continue

                if not top_started:
                    if ch.isspace():
                        continue

                    if ch != "[":
                        raise ValueError(
                            f"{path}: first JSON value is not an array"
                        )

                    top_started = True
                    square_depth = 1
                    continue

                # Detect a top-level value before changing nesting.
                if (
                    square_depth == 1
                    and brace_depth == 0
                    and not ch.isspace()
                    and ch not in ",]"
                ):
                    top_has_value = True

                if ch == "[":
                    square_depth += 1

                elif ch == "]":
                    square_depth -= 1

                    if square_depth < 0:
                        raise ValueError(
                            f"{path}: invalid JSON nesting"
                        )

                    # First top-level array has closed.
                    if square_depth == 0:
                        if not top_has_value:
                            return 0
                        return separators + 1

                elif ch == "{":
                    brace_depth += 1

                elif ch == "}":
                    brace_depth -= 1

                    if brace_depth < 0:
                        raise ValueError(
                            f"{path}: invalid JSON object nesting"
                        )

                elif (
                    ch == ","
                    and square_depth == 1
                    and brace_depth == 0
                ):
                    separators += 1

    if not top_started:
        raise ValueError(
            f"{path}: no JSON value found"
        )

    raise ValueError(
        f"{path}: top-level JSON array was not closed"
    )


def count_json_city(
    city: str,
) -> tuple[int, int]:
    files = _expand_trajectory_files(
        CONFIG_FILES[city]
    )

    print()
    print("=" * 72)
    print(
        f"Counting {city}: "
        f"{len(files)} trajectory file(s)"
    )
    print("=" * 72)

    total = 0

    for i, path in enumerate(
        files,
        1,
    ):
        n = count_top_level_json_array(
            path
        )
        total += n

        print(
            f"[{i}/{len(files)}] "
            f"{path.name}: {n} "
            f"| cumulative={total}"
        )

    print(
        f"{city} TOTAL = {total}"
    )

    return total, len(files)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(
                limit
            )
            return
        except OverflowError:
            limit //= 10


def count_beijing_csv() -> int:
    if not BEIJING_FILE.exists():
        raise FileNotFoundError(
            BEIJING_FILE
        )

    _raise_csv_field_limit()

    print()
    print("=" * 72)
    print("Counting beijing")
    print("=" * 72)
    print(BEIJING_FILE)

    total = 0

    with BEIJING_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.reader(f)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(
                f"{BEIJING_FILE}: empty CSV"
            )

        if not header:
            raise ValueError(
                f"{BEIJING_FILE}: missing header"
            )

        for total, _ in enumerate(
            reader,
            1,
        ):
            if total % 5000 == 0:
                print(
                    f"beijing processed: {total}"
                )

    print(
        f"beijing TOTAL = {total}"
    )

    return total


def cutoff(
    total: int,
    percent: int,
) -> int:
    if percent == 100:
        return total

    # Same policy used in the scaling experiment:
    # floor(total * percent / 100).
    return (
        total
        *
        percent
        //
        100
    )


def save_results(
    totals: dict[str, int],
    source_file_counts: dict[str, int],
) -> None:
    percents = (
        20,
        40,
        60,
        80,
        100,
    )

    payload = {
        "totals": totals,
        "cutoffs": {
            city: {
                str(p): cutoff(
                    total,
                    p,
                )
                for p in percents
            }
            for city, total
            in totals.items()
        },
        "source_file_counts":
        source_file_counts,
        "definition": (
            "Each scaling point uses the first N trajectories "
            "in the original deterministic input order. "
            "20% subset is contained in 40%, then 60%, 80%, 100%."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "city",
                "total_trajectories",
                "20%",
                "40%",
                "60%",
                "80%",
                "100%",
            ]
        )

        for city in (
            "chengdu",
            "xian",
            "beijing",
        ):
            total = totals[city]

            writer.writerow(
                [
                    city,
                    total,
                    *[
                        cutoff(
                            total,
                            p,
                        )
                        for p in (
                            20,
                            40,
                            60,
                            80,
                            100,
                        )
                    ],
                ]
            )

    print()
    print("=" * 72)
    print("FINAL COUNTS")
    print("=" * 72)

    for city in (
        "chengdu",
        "xian",
        "beijing",
    ):
        total = totals[city]

        values = [
            cutoff(
                total,
                p,
            )
            for p in (
                20,
                40,
                60,
                80,
                100,
            )
        ]

        print(
            f"{city:8s}: total={total}, "
            f"20/40/60/80/100%={values}"
        )

    print()
    print(
        "JSON:",
        OUTPUT_JSON,
    )
    print(
        "CSV :",
        OUTPUT_CSV,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        choices=[
            "all",
            "chengdu",
            "xian",
            "beijing",
        ],
        default="all",
    )

    args = parser.parse_args()

    # If only one city is requested, print it without requiring
    # the other datasets to exist.
    if args.city != "all":

        if args.city == "beijing":
            total = count_beijing_csv()
            print(
                f"\nbeijing total={total}"
            )
        else:
            total, _ = count_json_city(
                args.city
            )
            print(
                f"\n{args.city} total={total}"
            )

        print(
            "cutoffs:",
            {
                p: cutoff(
                    total,
                    p,
                )
                for p in (
                    20,
                    40,
                    60,
                    80,
                    100,
                )
            },
        )
        return

    totals: dict[str, int] = {}
    source_file_counts: dict[str, int] = {}

    for city in (
        "chengdu",
        "xian",
    ):
        (
            totals[city],
            source_file_counts[city],
        ) = count_json_city(city)

    totals["beijing"] = (
        count_beijing_csv()
    )
    source_file_counts[
        "beijing"
    ] = 1

    save_results(
        totals,
        source_file_counts,
    )


if __name__ == "__main__":
    main()
