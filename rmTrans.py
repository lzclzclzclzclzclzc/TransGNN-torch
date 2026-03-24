import argparse
import csv
import locale
import os
import re
import subprocess
import sys
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"Test:\s*Recall\s*=\s*([0-9eE+\-\.]+),\s*NDCG\s*=\s*([0-9eE+\-\.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TransGNN without Transformer (rmTrans) for block_num=1..4, each repeated 10 times."
    )
    parser.add_argument("--repeats", type=int, default=10, help="runs per layer")
    parser.add_argument("--layers", type=str, default="1,2,3,4", help="comma-separated layer list")
    parser.add_argument("--epoch", type=int, default=40, help="training epochs for each run")
    parser.add_argument("--data", type=str, default="yelp", help="dataset name for Main.py")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id passed to Main.py")
    parser.add_argument("--output", type=str, default="rmTrans.csv", help="output csv file name")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="python executable used to launch Main.py",
    )
    return parser.parse_args()


def ensure_csv_header(csv_path: Path):
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "repeat", "recall", "ndcg"])


def load_finished(csv_path: Path):
    finished = set()
    if not csv_path.exists():
        return finished

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                layer = int(str(row[0]).strip())
                repeat_idx = int(str(row[1]).strip())
            except ValueError:
                continue
            finished.add((layer, repeat_idx))
    return finished


def append_result(csv_path: Path, layer: int, repeat_idx: int, recall, ndcg):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([layer, repeat_idx, recall, ndcg])
        f.flush()
        os.fsync(f.fileno())


def parse_metrics(output_text: str):
    matches = list(METRIC_PATTERN.finditer(output_text))
    if not matches:
        return "", ""
    recall_str, ndcg_str = matches[-1].groups()
    return float(recall_str), float(ndcg_str)


def run_one(project_root: Path, python_bin: str, layer: int, repeat_idx: int, epoch: int, data: str, gpu: str):
    save_name = f"rmTrans_l{layer}_r{repeat_idx}"
    cmd = [
        python_bin,
        "-u",
        "Main.py",
        "--block_num",
        str(layer),
        "-rmTrans",
        "true",
        "--save_path",
        save_name,
        "--epoch",
        str(epoch),
        "--data",
        data,
        "--gpu",
        gpu,
    ]

    encoding = locale.getpreferredencoding(False) or "utf-8"
    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=encoding,
        errors="ignore",
        bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    process.wait()

    output_text = "".join(output_lines)
    recall, ndcg = parse_metrics(output_text)
    return process.returncode, recall, ndcg


def main():
    args = parse_args()
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    root = Path(__file__).resolve().parent
    csv_path = (root / args.output).resolve()
    ensure_csv_header(csv_path)
    finished = load_finished(csv_path)

    total = len(layers) * args.repeats
    done = len([1 for l in layers for r in range(1, args.repeats + 1) if (l, r) in finished])
    print(f"CSV: {csv_path}")
    print(f"Resume state: {done}/{total} already finished.")

    cur = 0
    for layer in layers:
        for repeat_idx in range(1, args.repeats + 1):
            cur += 1
            if (layer, repeat_idx) in finished:
                print(f"[{cur}/{total}] layer={layer}, repeat={repeat_idx} -> skip (already in csv)")
                continue

            print(f"[{cur}/{total}] layer={layer}, repeat={repeat_idx} -> running")
            code, recall, ndcg = run_one(
                project_root=root,
                python_bin=args.python,
                layer=layer,
                repeat_idx=repeat_idx,
                epoch=args.epoch,
                data=args.data,
                gpu=args.gpu,
            )

            append_result(csv_path, layer, repeat_idx, recall, ndcg)
            finished.add((layer, repeat_idx))

            if code != 0:
                print(f"run failed with return code={code}, row saved with parsed metrics (may be empty).")
            else:
                print(f"saved: layer={layer}, repeat={repeat_idx}, recall={recall}, ndcg={ndcg}")

    print(f"All tasks processed. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
