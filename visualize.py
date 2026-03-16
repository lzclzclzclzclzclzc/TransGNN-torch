import matplotlib.pyplot as plt
import collections

def plot_from_file(filename='Params_output.csv'):
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: {'Recall': [], 'NDCG': []}))
    
    # 1. 解析文件
    print(f"📖 正在读取 {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            next(f) # 跳过表头
            for line in f:
                p_name, val, trial, recall, ndcg = line.strip().split(',')
                data[p_name][val]['Recall'].append(float(recall))
                data[p_name][val]['NDCG'].append(float(ndcg))
    except FileNotFoundError:
        print("❌ 错误：找不到数据文件，请先运行 test.py")
        return

    # 2. 绘图
    for param_name, values_dict in data.items():
        print(f"🎨 正在生成 {param_name} 的图表...")
        
        # 排序取值（确保横坐标逻辑正确，如 16, 32, 64...）
        sorted_vals = sorted(values_dict.keys(), key=lambda x: float(x))
        labels = [str(v) for v in sorted_vals]
        
        recall_data = [values_dict[v]['Recall'] for v in sorted_vals]
        ndcg_data = [values_dict[v]['NDCG'] for v in sorted_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        
        # Recall 子图
        b1 = ax1.boxplot(recall_data, labels=labels, patch_artist=True, 
                         boxprops=dict(facecolor='#AED6F1'))
        ax1.set_title(f'Recall@40 vs {param_name}', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        
        # NDCG 子图
        b2 = ax2.boxplot(ndcg_data, labels=labels, patch_artist=True, 
                         boxprops=dict(facecolor='#ABEBC6'))
        ax2.set_title(f'NDCG@40 vs {param_name}', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        plt.suptitle(f'Parameter Sensitivity Analysis: {param_name}', fontsize=15)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = f"BoxPlot_{param_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" ✅ 已保存: {save_path}")

if __name__ == "__main__":
    plot_from_file()