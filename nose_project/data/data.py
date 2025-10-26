import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
feature_df = pd.read_csv(r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_health_test_std.csv")

index_to_nose = feature_df['speed_std']
index_to_nose_label = feature_df['label']

data_class_0 = index_to_nose[index_to_nose_label == 0]
data_class_1 = index_to_nose[index_to_nose_label == 1]

print(data_class_0)
print(data_class_1)



# 计算类别0的四分位数
Q1_class_0 = np.percentile(data_class_0, 25)
Q2_class_0 = np.median(data_class_0)
Q3_class_0 = np.percentile(data_class_0, 75)
IQR_class_0 = Q3_class_0 - Q1_class_0

lower_bound_class_0 = Q1_class_0 - 1.5 * IQR_class_0
upper_bound_class_0 = Q3_class_0 + 1.5 * IQR_class_0

threshold_Q1_class_0 = Q1_class_0
predicted_labels_Q1_0 = np.where(index_to_nose > threshold_Q1_class_0, 1, 0)
accuracy_Q1_0 = accuracy_score(index_to_nose_label, predicted_labels_Q1_0)

threshold_Q2_class_0 = Q2_class_0
predicted_labels_Q2_0 = np.where(index_to_nose > threshold_Q2_class_0, 1, 0)
accuracy_Q2_0 = accuracy_score(index_to_nose_label, predicted_labels_Q2_0)

threshold_Q3_class_0 = Q3_class_0
predicted_labels_Q3_0 = np.where(index_to_nose > threshold_Q3_class_0, 1, 0)
accuracy_Q3_0 = accuracy_score(index_to_nose_label, predicted_labels_Q3_0)

threshold_upper_bound_class_0 = upper_bound_class_0
predicted_labels_upper_bound_0 = np.where(index_to_nose > threshold_upper_bound_class_0, 1, 0)
accuracy_upper_bound_0 = accuracy_score(index_to_nose_label, predicted_labels_upper_bound_0)

threshold_lower_bound_class_0 = lower_bound_class_0
predicted_labels_lower_bound_0 = np.where(index_to_nose > threshold_lower_bound_class_0, 1, 0)
accuracy_lower_bound_0 = accuracy_score(index_to_nose_label, predicted_labels_lower_bound_0)

# 计算类别1的四分位数
Q1_class_1 = np.percentile(data_class_1, 25)
Q2_class_1 = np.median(data_class_1)
Q3_class_1 = np.percentile(data_class_1, 75)
IQR_class_1 = Q3_class_1 - Q1_class_1

lower_bound_class_1 = Q1_class_1 - 1.5 * IQR_class_1
upper_bound_class_1 = Q3_class_1 + 1.5 * IQR_class_1

threshold_Q1_class_1 = Q1_class_1
predicted_labels_Q1_1 = np.where(index_to_nose > threshold_Q1_class_1, 1, 0)
accuracy_Q1_1 = accuracy_score(index_to_nose_label, predicted_labels_Q1_1)

threshold_Q2_class_1 = Q2_class_1
predicted_labels_Q2_1 = np.where(index_to_nose > threshold_Q2_class_1, 1, 0)
accuracy_Q2_1 = accuracy_score(index_to_nose_label, predicted_labels_Q2_1)

threshold_Q3_class_1 = Q3_class_1
predicted_labels_Q3_1 = np.where(index_to_nose > threshold_Q3_class_1, 1, 0)
accuracy_Q3_1 = accuracy_score(index_to_nose_label, predicted_labels_Q3_1)

threshold_upper_bound_class_1 = upper_bound_class_1
predicted_labels_upper_bound_1 = np.where(index_to_nose > threshold_upper_bound_class_1, 1, 0)
accuracy_upper_bound_1 = accuracy_score(index_to_nose_label, predicted_labels_upper_bound_1)

threshold_lower_bound_class_1 = lower_bound_class_1
predicted_labels_lower_bound_1 = np.where(index_to_nose > threshold_lower_bound_class_1, 1, 0)
accuracy_lower_bound_1 = accuracy_score(index_to_nose_label, predicted_labels_lower_bound_1)

accuracy_threshold = {
    'Threshold': ['Q1_0', 'Q2_0', 'Q3_0', 'upper_bound_0','lower_bound_0', 
                  'Q1_1', 'Q2_1', 'Q3_1', 'upper_bound_1','lower_bound_1'],
    'threshold': [Q1_class_0, Q2_class_0, Q3_class_0, upper_bound_class_0, lower_bound_class_0, 
                  Q1_class_1, Q2_class_1, Q3_class_1, upper_bound_class_1, lower_bound_class_1],
    'accuracy_value': [accuracy_Q1_0, accuracy_Q2_0, accuracy_Q3_0, accuracy_upper_bound_0, accuracy_lower_bound_0, 
                       accuracy_Q1_1, accuracy_Q2_1, accuracy_Q3_1, accuracy_upper_bound_1, accuracy_lower_bound_1]
}
accuracy_threshold_df = pd.DataFrame(accuracy_threshold)
print(accuracy_threshold_df)
# accuracy_threshold_df.to_csv(r"output\trajectory\distance.csv")

# 绘制小提琴图

# 1. 创建类别标签
class_0_labels = ['T'] * len(data_class_0)  # 创建 Class 0 标签
class_1_labels = ['F'] * len(data_class_1)  # 创建 Class 1 标签

# 2. 合并数据和标签
data_combined = np.concatenate([data_class_0, data_class_1])  # 合并数据
labels_combined = np.concatenate([class_0_labels, class_1_labels])  # 合并类别标签

# 3. 创建一个 DataFrame 包含数据和类别标签
df_violin = pd.DataFrame({
    'speed_std': data_combined,
    'Class': labels_combined
})

# 4. 使用 Seaborn 绘制小提琴图
plt.figure(figsize=(7,5))
plt.rc('font', size = 13, weight = 'demibold', family = 'Times New Roman')
sns.violinplot(x='Class', y='speed_std', data=df_violin,inner= None, hue='Class', linewidth=1)
plt.title('Violin Plot of speed_std for True and False', fontsize=16, fontweight = 'bold')

plt.xlabel('Class',fontsize = 13, fontweight = 'demibold')
plt.ylabel('speed_std',fontsize = 13, fontweight = 'demibold')

plt.savefig(r"output\directivity\violin_speed_std_svg.svg")
plt.show()

# # 使用 Seaborn 绘制箱线图
# # ✅ 手动加载支持中文的字体文件（以 SimHei 为例）
# font_path = "C:/Windows/Fonts/SimHei.ttf"  # 黑体
# my_font = font_manager.FontProperties(fname=font_path)

# # ✅ 防止负号变方块
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(6, 5))
# plt.rc('font', size=13, weight='demibold', family='Times New Roman')
# sns.boxplot(x='Class', y='distance', data=df_violin, width=0.5,
#             fliersize=4, linewidth=1.5, palette="pastel")

# plt.title('T类与F类数据分布箱线图', fontproperties=my_font, fontsize=16, fontweight='bold')
# plt.xlabel('类别',fontproperties=my_font, fontsize=15, fontweight='demibold')
# plt.ylabel('距离', fontproperties=my_font,fontsize=15, fontweight='demibold')

# # 保存为 SVG 文件
# plt.savefig(r"output\directivity\box_distance_svg.svg")
# plt.show()