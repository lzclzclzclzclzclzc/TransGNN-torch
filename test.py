import os
import subprocess
import pickle
import sys
import re

# =================配置区=================
NUM_TRIALS = 10
TEST_EPOCHS = 40
DATASET = 'yelp'
GPU_ID = '0'

PARAMS_GRID = {
    'latdim': [16, 32, 64, 128],
    'block_num': [1, 2, 3, 4],
    'num_head': [1, 2, 4, 8],
    'edgeSampRate': [0.1, 0.3, 0.5, 0.7]
}
# ========================================


def load_finished_tasks(csv_file):
    """读取CSV中已经完成的实验"""
    finished = set()

    if not os.path.exists(csv_file):
        return finished

    with open(csv_file, 'r', encoding='utf-8') as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            param = parts[0]
            val = parts[1]
            trial = int(parts[2])

            finished.add((param, val, trial))

    return finished


def run_test():

    output_file = 'Params_output.csv'

    # 初始化CSV
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Param_Name,Value,Trial,Recall,NDCG\n")

    finished_tasks = load_finished_tasks(output_file)

    for param, values in PARAMS_GRID.items():

        print(f"\n开始分析参数: 【{param}】")

        for val in values:

            for i in range(NUM_TRIALS):

                # 只看CSV是否存在
                if (param, str(val), i) in finished_tasks:
                    print(f"  > 跳过 {param}={val} (Trial {i+1}) - CSV已有结果")
                    continue

                save_name = f"exp_{param}_{val}_t{i}"
                his_path = f"History/{save_name}.his"

                cmd = [
                    sys.executable, "-u", "Main.py",
                    f"--{param}", str(val),
                    "--save_path", save_name,
                    "--epoch", str(TEST_EPOCHS),
                    "--data", DATASET,
                    "--gpu", GPU_ID
                ]

                print(f"\n  > 开始运行: {param}={val} | Trial {i+1}/{NUM_TRIALS}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='gbk'
                )

                current_epoch = 0

                for line in process.stdout:

                    # 打印真实输出
                    print(line, end="")

                    match = re.search(r'Epoch (\d+)/(\d+)', line)

                    if match:
                        current_epoch = match.group(1)
                        total_epoch = match.group(2)

                        sys.stdout.write(
                            f"\r  > {param}={val:<5} | Trial {i+1}/{NUM_TRIALS} | Epoch [{current_epoch}/{total_epoch}]"
                        )
                        sys.stdout.flush()

                process.wait()

                # 如果程序崩溃
                if process.returncode != 0:
                    print(f"\n❌ Main.py 运行失败 (returncode={process.returncode})")
                    continue

                # 尝试读取his
                if not os.path.exists(his_path):
                    print(f"\n❌ 未找到结果文件 {his_path}")
                    continue

                with open(his_path, 'rb') as hf:

                    data = pickle.load(hf)

                    r = data['TestRecall'][-1] if data['TestRecall'] else 0
                    n = data['TestNDCG'][-1] if data['TestNDCG'] else 0

                # 写入CSV
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{param},{val},{i},{r:.6f},{n:.6f}\n")

                finished_tasks.add((param, str(val), i))

                print(f"\n✓ 完成 {param}={val} Trial {i+1} | Recall {r:.4f} | NDCG {n:.4f}")

    print(f"\n✅ 所有实验完成！结果保存在 {output_file}")


if __name__ == "__main__":
    run_test()