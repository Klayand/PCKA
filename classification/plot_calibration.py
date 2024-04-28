import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
import torch

# 加载数据
data = np.load('/home/project/FAKD/classification/result/imagenet_traj_point_8.npz')
logits = data['output']
targets = data['target']

# 将logits转换为概率（使用softmax函数）
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

probs = softmax(logits)
targets = (torch.from_numpy(probs).argmax(1) == torch.from_numpy(targets).int()).int().numpy()
predicted_probs = probs.max(axis=1)

# 计算可靠性直方图
prob_true, prob_pred = calibration_curve(targets, predicted_probs, n_bins=40, strategy='uniform')

# 绘制直方图
bin_width = 0.015  # bin宽度
bin_edges = np.linspace(0, 1, 41)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

colors = ['#FFC75F', '#00C9A7', '#d1c7b7', '#33a3dc']

plt.figure(figsize=(10, 8))
for i in range(len(prob_true)):
    if prob_true[i] > bin_centers[i]:
        plt.bar(bin_centers[i], bin_centers[i], width=bin_width*0.9, color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.bar(bin_centers[i], prob_true[i] - bin_centers[i], bottom=bin_centers[i], width=bin_width*0.9, color=colors[3], alpha=0.8, edgecolor='black', linewidth=0.5)
    else:
        plt.bar(bin_centers[i], prob_true[i], width=bin_width*0.9, color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)

plt.fill_between([0, 1], [0, 1], color='#FF6347', alpha=0.2)  # 淡红棕色填充
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
plt.xlabel("Mean predicted probability", fontsize=14)
plt.ylabel("Fraction of positives", fontsize=14)
plt.title("Reliability Histogram", fontsize=16)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
plt.savefig('/home/project/FAKD/classification/result/imagenet_traj_point_8.png')
plt.show()
