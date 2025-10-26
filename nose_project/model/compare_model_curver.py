from math import pi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.ticker import MaxNLocator

# 读取数据
feature_df = pd.read_csv(r"dataset\trajectory_test.csv")

# 定义特征组合
MCCR_MSA_MTA = ['mean_curvetrue_change_rate','mean_slope_abs', 'mean_triangle_area']
X = feature_df[MCCR_MSA_MTA]
y = feature_df['label']

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义分类器
classifiers = {
    "SVM (Best)": joblib.load(r'output\showPhoto\MCCR_MSA_MTA_model.pkl'),  # 加载最佳 SVM 模型
    "KNN": KNeighborsClassifier(n_neighbors=10),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42)
}

# 存储分类器的结果
results = []
roc_data = {}  # Store ROC data
pr_data = {}  # Store PR data
for name, clf in classifiers.items():
    # 训练分类器
    clf.fit(X_train, y_train)
    # 预测
    y_pred = clf.predict(X_test)
    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # 存储结果
    results.append({
        'Classifier': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    # 处理概率或决策函数
    if hasattr(clf, "predict_proba"):  # 检查是否支持 `predict_proba`
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):  # 如果支持 `decision_function`
        y_pred_proba = clf.decision_function(X_test)
        # 将决策分数缩放到 [0, 1] 范围
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
    else:
        print(f"{name} does not support probability or decision function.")
        continue  # 跳过不支持的分类器

    # 绘制ROC PR 曲线 预测概率
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # 修正：直接使用 `y_pred_proba`
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)  # 修正：直接使用 `y_pred_proba`
    pr_auc = auc(recall, precision)
    pr_data[name] = (recall, precision, pr_auc)
# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 绘制雷达图
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
num_vars = len(categories)

# 设置雷达图的角度
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # 闭合雷达图

plt.figure(figsize=(6, 6))
plt.rc('font', size=13, family="Times New Roman", weight = 'bold')

# 绘制每个分类器的雷达图
for i, row in results_df.iterrows():
    values = row[categories].tolist()
    values += values[:1]  # 闭合雷达图
    plt.polar(angles, values, marker='o', label=row['Classifier'])
    plt.fill(angles, values, alpha=0.25)

# 隐藏角度标注（默认的刻度）
plt.xticks([], [])  # 去除默认角度刻度

# 设置标签
for i, label in enumerate(categories):
    angle = angles[i]
    if label in ['Recall', 'Accuracy']:
        rotation = 90 if angle == 0 or angle == pi else -90
        plt.text(
            angle, 1.1, label, fontsize=16, fontweight='bold', 
            ha='center', va='center', rotation=rotation
        )
    else:
        plt.text(
            angle, 1.1, label, fontsize=16, fontweight='bold',
            ha='center', va='center', rotation=0
        )

# 设置刻度和范围
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14, fontweight = 'bold')
plt.ylim(0, 1)

# 添加标题和图例
plt.title("Classifier Metric Comparison", fontsize=18, fontweight='bold',y = 1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)
plt.tight_layout()

# 保存图片
plt.savefig("output/Classifier_Metrics_Radar.svg")
plt.close()

# Plot overall ROC curve
plt.rc('font', size=16, weight='normal', family="Times New Roman")
plt.figure(figsize=(11.9, 5.1))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
# plt.plot([0, 1], [0, 1], 'k--'``)
plt.title("ROC Curve Comparison", fontsize = 18, fontweight = 'black')
plt.xlabel("False Positive Rate", fontsize = 17, fontweight = 'black')
plt.ylabel("True Positive Rate", fontsize = 17, fontweight = 'black')
plt.xticks( fontsize = 16, fontweight = 'normal')
plt.yticks(fontsize = 16, fontweight = 'normal')

plt.legend(loc="lower right")
# plt.grid()
plt.savefig("output/Diff_Model_ROC_Curve_resize.svg",dpi = 300, pad_inches=0.001)
plt.show()
plt.close()

# # Plot overall PR curve
# plt.rc('font', size=14, weight='bold', family="Times New Roman")
# plt.figure(figsize=(6, 5))
# for name, (recall, precision, pr_auc) in pr_data.items():
#     plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")
# # plt.plot([0, 1], [1,0], color='gray', lw=2, linestyle='--')
# plt.title("PR Curve Comparison", fontsize = 16, fontweight = 'bold')
# plt.xlabel("Recall", fontsize = 14, fontweight = 'bold')
# plt.ylabel("Precision", fontsize = 14, fontweight = 'bold')
# plt.legend(loc="lower left")
# plt.grid()
# plt.savefig("output/Diff_Model_PR_Curve.png")
# plt.show()
# plt.close()
