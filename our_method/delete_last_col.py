from pathlib import Path


# ===== Modify here =====
INPUT_FILE = Path(r"E:\MMR_Trajectory_range\xian_edges.txt")
OUTPUT_FILE = Path(r"/xian_edges.txt")


def remove_last_column(input_file: Path, output_file: Path) -> None:
    """
    Remove the last column from each line of the TXT file.
    Columns are separated by any whitespace characters,
    such as spaces or tabs.
    """

    with input_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8", newline="") as fout:

        line_count = 0

        for line in fin:
            stripped = line.strip()

            if not stripped:
                fout.write("\n")
                continue

            columns = stripped.split()

            if len(columns) >= 2:
                columns = columns[:-1]

            fout.write(" ".join(columns) + "\n")
            line_count += 1

    print("Processing completed")
    print("Input file:", input_file)
    print("Output file:", output_file)
    print("Number of processed lines:", line_count)


if __name__ == "__main__":
    remove_last_column(
        INPUT_FILE,
        OUTPUT_FILE,
    )