import numpy as np
import pandas as pd
import joblib
import seaborn as sns
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, RocCurveDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


feature_df = pd.read_csv(r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_test.csv")

MCCR_MSA = [ 'mean_curvetrue_change_rate',  'mean_slope_abs']
MCCR_MTA = ['mean_curvetrue_change_rate', 'mean_triangle_area']
MSA_MTA = ['mean_slope_abs', 'mean_triangle_area']
MCCR_MSA_MTA = ['mean_curvetrue_change_rate','mean_slope_abs', 'mean_triangle_area']
RS_MCCR_MSA_MTA = ['R_squared','mean_curvetrue_change_rate','mean_slope_abs', 'mean_triangle_area']
RS_MSA = ['R_squared','mean_slope_abs']

features = {
    "MCCR_MSA": MCCR_MSA,
    "MCCR_MTA": MCCR_MTA,
    "MSA_MTA": MSA_MTA,
    "MCCR_MSA_MTA": MCCR_MSA_MTA,
    "RS_MCCR_MSA_MTA": RS_MCCR_MSA_MTA,
    "RS_MSA":RS_MSA
}

results = []
roc_data = {}  # Store ROC data
pr_data = {}  # Store PR data
for name, feature in features.items():
    X = feature_df[feature]
    y = feature_df['label']

    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"output\model_pkl\{name}_scaler_rs.pkl")

    # 2. 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    print("训练集分布:", Counter(y_train))
    print("测试集分布:", Counter(y_test))

    # 3. 网格搜索
    param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }

    model = svm.SVC(kernel ='linear', probability = True)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"{feature} Best Parameters found:", grid_search.best_params_)
    print(f"{feature} Beat cross-validation f1 score:", grid_search.best_score_)

    best_svm = grid_search.best_estimator_
    
    # 绘制ROC PR 曲线 预测概率
    y_pred_proba = best_svm.predict_proba(X_test)
    fpr, tpr , _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    pr_auc = auc(recall, precision)
    pr_data[name] = (recall, precision, pr_auc)

    # 绘制对比图
    y_pred = best_svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'best_svm': best_svm,
        'Feature Combination': name,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })


# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 绘制柱状图对比
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df['Feature Combination'])) * 0.25  # x轴位置
width = 0.05  # 每个柱子的宽度

# 设置字体
plt.rc('font', size=18, weight='black', family="Times New Roman")

# 创建图形
plt.figure(figsize=(10.5, 4.5))

colors = sns.color_palette('colorblind')
# 绘制柱状图
for i, (metric,color) in enumerate(zip(metrics,colors)):
    plt.bar(
        x + i * width,
        results_df[metric],
        width,
        label=metric,
        color=color
    )

# 添加标签、标题和图例
plt.xlabel('Feature Combination', fontsize=17, fontweight = 'black')
plt.ylabel('Metric Score', fontsize=17, fontweight = 'black')
plt.title('Comparison of Metrics Across Feature Combinations', fontsize=18, fontweight = 'black')
plt.xticks(x + width * (len(metrics) - 1) / 2, results_df['Feature Combination'], fontsize=16, rotation=8, fontweight = 'normal')
plt.yticks(fontsize = 16, fontweight = 'normal')
plt.ylim([0, 1.2])
plt.legend(loc= 'lower right',fontsize=16)

# 在柱状图顶部显示值
for i, metric in enumerate(metrics):
    for j, value in enumerate(results_df[metric]):
        plt.text(
            x[j] + i * width,
            value + 0.02,
            f"{value:.4f}",
            ha='center',
            fontsize=16,
            rotation=90
            
        )

# 调整布局并保存图片
plt.tight_layout()
plt.savefig("output\Feature_Combination_Metrics_Comparison_resize_rs.svg")
plt.close()
# plt.show()

# Plot overall ROC curve
plt.rc('font', size=16, weight='normal', family="Times New Roman")
plt.figure(figsize=(11.9, 5.1))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
# plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison", fontsize = 18, fontweight = 'black')
plt.xlabel("False Positive Rate", fontsize = 17, fontweight = 'black')
plt.ylabel("True Positive Rate", fontsize = 17, fontweight = 'black')
plt.xticks( fontsize = 16, fontweight = 'normal')
plt.yticks(fontsize = 16, fontweight = 'normal')
plt.legend(loc="lower right")
# plt.grid()
plt.savefig("output/Overall_ROC_Curve_resize_rs.svg")
# plt.show()
plt.close()

# Plot overall PR curve
plt.rc('font', size=12, weight='bold', family="Times New Roman")
plt.figure(figsize=(7, 3))
for name, (recall, precision, pr_auc) in pr_data.items():
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")
# plt.plot([0, 1], [1,0], color='gray', lw=2, linestyle='--')
plt.title("PR Curve Comparison", fontsize = 16, fontweight = 'bold')
plt.xlabel("Recall", fontsize = 14, fontweight = 'bold')
plt.ylabel("Precision", fontsize = 14, fontweight = 'bold')
plt.legend(loc="lower left")
# plt.grid()
plt.savefig("output/Overall_PR_Curve_resize_rs.svg")
# plt.show()
plt.close()

# best_svm = results[3]['best_svm']
# joblib.dump(best_svm, r'output\showPhoto\MCCR_MSA_MTA_model.pkl')
# print("Model saved to svm_model.pkl")



    