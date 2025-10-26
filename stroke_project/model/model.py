import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc,precision_recall_curve
import xgboost as xgb


df = pd.read_excel('D:\研究生文件记录\小论文资料汇总\实验代码\stroke_project\data.xlsx')
df.info()
df.drop('id', axis = 1, inplace = True)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders= {}
for col in cat_cols: 
    # Encode values in training set
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])
    label_encoders[col] = le

for col in cat_cols:
    print(f"特征列 '{col}' 的编码关系:")
    for i, class_label in enumerate(label_encoders[col].classes_):
        print(f"  {class_label} -> {i}")
df['labels'] = df['labels'].replace({0: 1, 1: 0})
print(df)
df.to_csv(r"output/encoded_data.csv")
class_counts = df['labels'].value_counts()
print(f"Class distribution: {class_counts}")

# """
# Nullpercent	    每列缺失值占总行数的百分比，用于评估缺失值严重性。
# NuniquePercent	每列唯一值占总行数的百分比，用于分析列的稀疏性或类别分布。
# dtype	        每列的数据类型，用于识别数值型或类别型数据等。
# Nuniques	    每列中唯一值的数量，显示该列中有多少不同的值（适用于类别型数据分析）。
# Nulls	        每列缺失值的数量，用于判断缺失数据是否需要处理。

# """
# null_percent = df.isnull().mean() * 100
# nunique_percent = df.nunique() / len(df) * 100
# dtypes = df.dtypes
# nuniques = df.nunique()

# data_cleaning_report = pd.DataFrame({
#     'Nullpercent': null_percent,
#     'NuniquePercent': nunique_percent,
#     'dtype': dtypes,
#     'Nuniques': nuniques,
#     'Nulls': df.isnull().sum()
# })
# print(f"Data report: {data_cleaning_report}")

X = df.drop('labels', axis=1)
y = df['labels']
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# STOME 类别平衡

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Class distribution before resampling: {y_train.value_counts()}")
print(f"Class distribution after resampling: {y_train_res.value_counts()}")

# 保存性能指标
all_output = {}
# 构建 RF 模型
rf0 = RandomForestClassifier(oob_score=False, n_estimators=100, random_state=10, max_depth=1)
rf0.fit(X_train_res, y_train_res)
rf0_score = cross_val_score(rf0, X_train_res, y_train_res, cv=10).mean()
print(f"rf0 score: {rf0_score}")

# RF 调参 

# param_grid = {
#             # 'n_estimators': range(1, 200, 1),
#             # 'max_depth': np.arange(1, 15, 1),
#             # 'max_features': np.arange(1, 20, 2), 
#             # 'min_samples_leaf': np.arange(1, 1+10, 1),
#             # 'min_samples_split': np.arange(2, 2+20, 1),
#             'criterion': ['gini', 'entropy']
#                 }
# rfc = RandomForestClassifier(oob_score=False, n_estimators=118, random_state=10, max_depth=7,max_features=3,
#                              min_samples_leaf=1,min_samples_split=7,criterion='gini')
# best_RF = GridSearchCV(rfc, param_grid, cv=10)
# best_RF.fit(X_train_res, y_train_res)
# print(best_RF.best_params_)
# print(f"best_RF score ：{best_RF.best_score_}")
# # score_pre = cross_val_score(best_RF, X_train_res, y_train_res, cv=10).mean()
# # print(f"best_RF score: {score_pre}")
best_RF = RandomForestClassifier(oob_score=False, n_estimators=118, random_state=10, max_depth=7,max_features=3,
                             min_samples_leaf=1,min_samples_split=7,criterion='gini')
joblib.dump(best_RF, r"output/best_RF.pkl")
best_RF.fit(X_train_res, y_train_res)
best_RF_score = cross_val_score(best_RF, X_train_res, y_train_res, cv=10).mean()
print(f"best_RF score: {best_RF_score}")

y_pred_best_rf = best_RF.predict(X_test)

# 计算性能指标
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
precision_best_rf = precision_score(y_test, y_pred_best_rf)
recall_best_rf = recall_score(y_test, y_pred_best_rf)
f1_best_rf = f1_score(y_test, y_pred_best_rf)
all_output['best_rf'] = {
        'accuracy': accuracy_best_rf,
        'precision': precision_best_rf,
        'recall': recall_best_rf,
        'F1_score': f1_best_rf  
}
print(f"""best_RF 性能指标：
      accuracy: {accuracy_best_rf},
      precision: {precision_best_rf},
      recall: {recall_best_rf},
      F1-score: {f1_best_rf}      

""")
print(rf0)
y_pred_rf0 = rf0.predict(X_test)
# 计算性能指标
accuracy_rf0= accuracy_score(y_test, y_pred_rf0)
precision_rf0 = precision_score(y_test, y_pred_rf0)
recall_rf0 = recall_score(y_test, y_pred_rf0)
f1_rf0 = f1_score(y_test, y_pred_rf0)
all_output['rf0'] = {
        'accuracy': accuracy_rf0,
        'precision': precision_rf0,
        'recall': recall_rf0,
        'F1_score': f1_rf0  
}
print(f"""初始RF 性能指标：
      accuracy: {accuracy_rf0},
      precision: {precision_rf0},
      recall: {recall_rf0},
      F1-score: {f1_rf0}      

""")

# 绘制 ROC
roc_data = {}
pr_data = {}

name = 'RF_Origin'
y_proba_rf0 = rf0.predict_proba(X_test)[:, 1]
fpr_rf0, tpr_rf0, _ = roc_curve(y_test, y_proba_rf0)
roc_auc_rf0 = auc(fpr_rf0, tpr_rf0)
print(f"rf0 AUC:{roc_auc_rf0}")
roc_data[name] = (fpr_rf0, tpr_rf0, roc_auc_rf0)

precision_rf0, recall_rf0, _ = precision_recall_curve(y_test, y_proba_rf0)
pr_auc_rf0 = auc(recall_rf0, precision_rf0)
pr_data[name] = ( recall_rf0, precision_rf0, pr_auc_rf0)

name = 'RF_best'
y_proba_best_RF = best_RF.predict_proba(X_test)[:, 1]
fpr_best_RF, tpr_best_RF, _ = roc_curve(y_test, y_proba_best_RF)
roc_auc_best_RF = auc(fpr_best_RF, tpr_best_RF)
roc_data[name] = (fpr_best_RF, tpr_best_RF, roc_auc_best_RF)

precision_best_RF, recall_best_RF, _ = precision_recall_curve(y_test, y_proba_best_RF)
pr_auc_best_RF = auc(recall_best_RF, precision_best_RF)
pr_data[name] = ( recall_best_RF, precision_best_RF, pr_auc_best_RF)

for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

plt.title("ROC Curve Comparison", fontsize = 16, fontweight = 'bold')
plt.xlabel("False Positive Rate", fontsize = 14, fontweight = 'bold')
plt.ylabel("True Positive Rate", fontsize = 14, fontweight = 'bold')

plt.legend(loc="lower right")
plt.grid()
plt.savefig("output/ROC Curves of Random Forest Before and After Tuning.svg")
# plt.show()
plt.close()

plt.rc('font', size=14, weight='bold', family="Times New Roman")
plt.figure(figsize=(6, 5))
for name, (recall, precision, pr_auc) in pr_data.items():
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")
# plt.plot([0, 1], [1,0], color='gray', lw=2, linestyle='--')
plt.title("PR Curve Comparison", fontsize = 16, fontweight = 'bold')
plt.xlabel("Recall", fontsize = 14, fontweight = 'bold')
plt.ylabel("Precision", fontsize = 14, fontweight = 'bold')
plt.legend(loc="lower left")
plt.grid()
plt.savefig("output/PR Curves of Random Forest Before and After Tuning.svg")
# plt.show()
plt.close()




# 构建XGBoost模型
xgb_0 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=5, min_child_weight=1)
xgb_0.fit(X_train_res, y_train_res)
score_pre_xgb = cross_val_score(xgb_0, X_train_res, y_train_res, cv=10).mean()
print(f"xgb0 score_pre: {score_pre_xgb}")

# # param_test = {
# #                 # 'n_estimators':range(1, 120, 1),
# #                 # 'max_depth': range(0, 10, 1),
# #                 # 'min_child_weight': range(0,7,1),
# #                 # 'gamma':[i/100.0 for i in range(0,100)],
# #                 # 'subsample':[i/20.0 for i in range(1,20)],
# #                 'colsample_bytree':[i/20.0 for i in range(1,20)],
# #                 # 'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
# #                 # 'learning_rate':[0, 0.001, 0.005, 0.01, 0.05,0.1,0.5,1]
                
# #             }

# # XGBR= xgb.XGBClassifier(n_estimators=89, learning_rate=1, max_depth=5, min_child_weight=2, gamma=0.14,subsample=0.65,reg_alpha=0.05)
# # best_xgb = GridSearchCV(XGBR, param_test, cv=10)
# # best_xgb.fit(X_train_res, y_train_res)
# # print(best_xgb.best_params_)
# # print("best_xgb score:", best_xgb.best_score_)

best_xgb= xgb.XGBClassifier(n_estimators=89, learning_rate=1, max_depth=5, min_child_weight=2, gamma=0.14,subsample=0.65,reg_alpha=0.05)
joblib.dump(best_RF, r"output/best_XGB.pkl")
best_xgb.fit(X_train_res, y_train_res)
best_xgb_score = cross_val_score(best_xgb, X_train_res, y_train_res, cv=10).mean()
print(f"best_RF score: {best_xgb_score}")
y_pred_best_xgb = best_xgb.predict(X_test)

# 计算性能指标
accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)
precision_best_xgb = precision_score(y_test, y_pred_best_xgb)
recall_best_xgb = recall_score(y_test, y_pred_best_xgb)
f1_best_xgb = f1_score(y_test, y_pred_best_xgb)
all_output['best_xgb'] = {
        'accuracy': accuracy_best_xgb,
        'precision': precision_best_xgb,
        'recall': recall_best_xgb,
        'F1_score': f1_best_xgb  
}
print(f"""best_xgb 性能指标：
      accuracy: {accuracy_best_xgb},
      precision: {precision_best_xgb},
      recall: {recall_best_xgb},
      F1-score: {f1_best_xgb}      

""")
print(xgb_0)
y_pred_xgb_0 = xgb_0.predict(X_test)
# 计算性能指标
accuracy_xgb0= accuracy_score(y_test, y_pred_xgb_0)
precision_xgb0 = precision_score(y_test, y_pred_xgb_0)
recall_xgb0 = recall_score(y_test, y_pred_xgb_0)
f1_xgb0 = f1_score(y_test, y_pred_xgb_0)
all_output['xgb0'] = {
        'accuracy': accuracy_xgb0,
        'precision': precision_xgb0,
        'recall': recall_xgb0,
        'F1_score': f1_xgb0  
}
print(f"""初始xgb 性能指标：
      accuracy: {accuracy_xgb0},
      precision: {precision_xgb0},
      recall: {recall_xgb0},
      F1-score: {f1_xgb0}      

""")
# 绘制 ROC
roc_data_xgb = {}
pr_data_xgb = {}

name = 'XGB_Origin'
y_proba_xgb0 = xgb_0.predict_proba(X_test)[:, 1]
fpr_xgb0, tpr_xgb0, _ = roc_curve(y_test, y_proba_xgb0)
roc_auc_xgb0 = auc(fpr_xgb0, tpr_xgb0)
print(f"xgb0 AUC:{roc_auc_xgb0}")
roc_data_xgb[name] = (fpr_xgb0, tpr_xgb0, roc_auc_xgb0)

precision_xgb0, recall_xgb0, _ = precision_recall_curve(y_test, y_proba_xgb0)
pr_auc_xgb0 = auc(recall_xgb0, precision_xgb0)
pr_data_xgb[name] = ( recall_xgb0, precision_xgb0, pr_auc_xgb0)

name = 'XGB_best'
y_proba_best_xgb = best_xgb.predict_proba(X_test)[:, 1]
fpr_best_xgb, tpr_best_xgb, _ = roc_curve(y_test, y_proba_best_xgb)
roc_auc_best_xgb = auc(fpr_best_xgb, tpr_best_xgb)
roc_data_xgb[name] = (fpr_best_xgb, tpr_best_xgb, roc_auc_best_xgb)

precision_best_xgb, recall_best_xgb, _ = precision_recall_curve(y_test, y_proba_best_xgb)
pr_auc_best_xgb = auc(recall_best_xgb, precision_best_xgb)
pr_data_xgb[name] = ( recall_best_xgb, precision_best_xgb, pr_auc_best_xgb)

for name, (fpr, tpr, roc_auc) in roc_data_xgb.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

plt.title("ROC Curve Comparison", fontsize = 16, fontweight = 'bold')
plt.xlabel("False Positive Rate", fontsize = 14, fontweight = 'bold')
plt.ylabel("True Positive Rate", fontsize = 14, fontweight = 'bold')

plt.legend(loc="lower right")
plt.grid()
plt.savefig("output/ROC Curves of XGBoost Before and After Tuning.png")
# plt.show()
plt.close()

plt.rc('font', size=14, weight='bold', family="Times New Roman")
plt.figure(figsize=(6, 5))
for name, (recall, precision, pr_auc) in pr_data.items():
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")
# plt.plot([0, 1], [1,0], color='gray', lw=2, linestyle='--')
plt.title("PR Curve Comparison", fontsize = 16, fontweight = 'bold')
plt.xlabel("Recall", fontsize = 14, fontweight = 'bold')
plt.ylabel("Precision", fontsize = 14, fontweight = 'bold')
plt.legend(loc="lower left")
plt.grid()
plt.savefig("output/PR Curves of XGBoost Before and After Tuning.png")
# plt.show()
plt.close()
# ----------------------------------



# 定义权重范围
weight_range = np.linspace(0, 1, 101)  # 权重从 0 到 1 以 0.01 为步长
best_a = 0
best_metric = -np.inf
best_score = None

# 用于存储每个权重的不同指标
metrics_dict = {}

# 对每个权重进行遍历
for a in weight_range:
    # 计算加权预测概率
    rf_prob = best_RF.predict_proba(X_test)[:, 1]
    xgb_prob = best_xgb.predict_proba(X_test)[:, 1]
    blended_prob = a * rf_prob + (1 - a) * xgb_prob
    
    # 根据阈值 0.5 生成预测结果
    blended_pred = (blended_prob >= 0.5).astype(int)
    
    # 计算各种指标
    accuracy = accuracy_score(y_test, blended_pred)
    precision = precision_score(y_test, blended_pred)
    recall = recall_score(y_test, blended_pred)
    f1 = f1_score(y_test, blended_pred)
    roc_auc = roc_auc_score(y_test, blended_prob)
    
    # 存储指标
    metrics_dict[a] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
  
    if roc_auc > best_metric:
        best_metric = roc_auc  # 若您想以 F1-score 为主，则改为 best_metric = f1
        best_a = a
        best_score = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }    

# 打印最佳结果
print(f"最佳权重 a: {best_a}")
print(f"基于roc_auc的最佳分数：")
print(f"准确率: {best_score['accuracy']}")
print(f"精确率: {best_score['precision']}")
print(f"召回率: {best_score['recall']}")
print(f"F1-Score: {best_score['f1_score']}")
print(f"ROC-AUC: {best_score['roc_auc']}")


# 使用最佳权重计算融合概率
best_blended_prob = best_a * best_RF.predict_proba(X_test)[:, 1] + (1 - best_a) * best_xgb.predict_proba(X_test)[:, 1]

# 优化分类阈值
threshold_range = np.linspace(0, 1, 101)
best_threshold = 0.5
best_f1 = -np.inf
threshold_metrics = {}

for threshold in threshold_range:
    blended_pred = (best_blended_prob >= threshold).astype(int)
    f1 = f1_score(y_test, blended_pred)
    
    threshold_metrics[threshold] = f1
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"最佳分类阈值: {best_threshold}")
print(f"对应的F1-Score: {best_f1}")

def drawRoc_three(name_file,roc_auc,fpr,tpr,roc_auc1,fpr1,tpr1,roc_auc2,fpr2,tpr2):
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(10) 
    plt.rc('font', family = 'Times New Roman', weight='bold')
    plt.figure(figsize=(11.9, 5.1))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='RF ROC curve (area = %0.4f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='b', lw=2, label='XGBoost ROC curve (area = %0.4f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='g', lw=2, label='XGBRForest ROC curve (area = %0.4f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1],lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.size'] = 12
    plt.xlabel('False Positive Rate',fontsize=17, fontweight = 'black')
    plt.ylabel('True Positive Rate',fontsize=17, fontweight = 'black')
    plt.xticks(np.arange(0, 1.05, 0.2), fontsize=16, fontweight='normal')  # 设置 x 轴刻度为从 0 到 10，间隔为 1
    plt.yticks(np.arange(0, 1.05, 0.2),  fontsize=16, fontweight='normal')  # 设置 y 轴刻度为从 -1 到 1，间隔为 0.2
    plt.title('ROC Curve',fontsize=18, fontweight = 'black')
    # plt.grid()
    plt.legend(loc="lower right",fontsize=16)
    
    plt.savefig(name_file)
    # plt.show()
# 定义融合模型类
class XGBRForest:
    def __init__(self, rf_model, xgb_model, a=0.5, threshold=0.5):
        """
        初始化融合模型。
        
        参数：
        - rf_model: 已训练的随机森林模型
        - xgb_model: 已训练的XGBoost模型
        - a: 随机森林模型的权重
        - threshold: 分类阈值
        """
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.a = a
        self.threshold = threshold
    
    def predict_proba(self, X):
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        blended_prob = self.a * rf_prob + (1 - self.a) * xgb_prob
        return np.vstack([1 - blended_prob, blended_prob]).T
    
    def predict(self, X):
        blended_prob = self.a * self.rf_model.predict_proba(X)[:, 1] + (1 - self.a) * self.xgb_model.predict_proba(X)[:, 1]
        return (blended_prob >= self.threshold).astype(int)

# 创建融合模型实例
xgbrforest_model = XGBRForest(
    rf_model=best_RF,
    xgb_model=best_xgb,
    a=0.69,
    threshold=0.43
)
joblib.dump(xgbrforest_model, r'output/xgbrforest.pkl')

# 计算预测概率
rf_prob = rf0.predict_proba(X_test)[:, 1]
xgb_prob = xgb_0.predict_proba(X_test)[:, 1]
xgbrforest_prob = xgbrforest_model.predict_proba(X_test)[:, 1]

# 计算ROC曲线和AUC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_xgb, tpr_xgb, _ =  roc_curve(y_test, xgb_prob)
roc_auc_xgb =  auc(fpr_xgb, tpr_xgb)

fpr_xgbrforest, tpr_xgbrforest, _ =  roc_curve(y_test, xgbrforest_prob)
roc_auc_xgbrforest =  auc(fpr_xgbrforest, tpr_xgbrforest)

# 调用绘图函数，保存为 'roc_comparison.png'
drawRoc_three(
    r'output/roc_comparison_three_resize_svg.svg',
    roc_auc_rf, fpr_rf, tpr_rf,
    roc_auc_xgb, fpr_xgb, tpr_xgb,
    roc_auc_xgbrforest, fpr_xgbrforest, tpr_xgbrforest
)

xgbrforest_prob = xgbrforest_model.predict_proba(X_test)[:, 1]
xgbrforest_pred = (xgbrforest_prob > 0.43).astype(int)

all_output['XGBRForest'] = {
    'accuracy': accuracy_score(y_test, xgbrforest_pred),
    'precision': precision_score(y_test, xgbrforest_pred),
    'recall': recall_score(y_test, xgbrforest_pred),
    'f1': f1_score(y_test, xgbrforest_pred)
}
print(all_output['XGBRForest'])
# with open('output/model_metrics.pkl', 'w') as json_file:
joblib.dump(all_output, 'output/model_metrics.pkl')

print("模型性能指标已保存为 'model_metrics.pkl'")


# y_test_pred = xgbrforest_model.predict(X_test)
# 生成混淆矩阵
cm = confusion_matrix(y_test, xgbrforest_pred)
# cm_percentage = cm / cm.sum(axis=1, keepdims=True) 
# 可视化混淆矩阵
plt.rc('font', family = "Times New Roman", weight='bold')
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, square=True,annot_kws={'size':15},
            xticklabels=['unstroke', 'stroke'], yticklabels=['unstroke', 'stroke']) 
plt.title('Confusion Matrix for XGBRForest', fontsize=15, fontweight = 'bold')
plt.xlabel('Predicted label', fontsize=14, fontweight = 'bold')
plt.ylabel('True label', fontsize=14, fontweight = 'bold')
plt.xticks(fontsize=14)  # 调整X轴刻度字体大小
plt.yticks(fontsize=14)  # 调整Y轴刻度字体大小
plt.savefig(r'output/xgbRF_confusion.svg') 
# plt.show()

y_test_pred = best_xgb.predict(X_test)
# 生成混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
# cm_percentage = cm / cm.sum(axis=1, keepdims=True) 
# 可视化混淆矩阵
plt.rc('font', family = 'Times New Roman', weight='bold')
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, square=True,
            xticklabels=['unstroke', 'stroke'], yticklabels=['unstroke', 'stroke']) 
# plt.title('Confusion Matrix for XGBRForest', fontsize=15)
plt.xlabel('Predicted label', fontsize=12, fontweight = 'bold')
plt.ylabel('True label', fontsize=12, fontweight = 'bold')
plt.xticks(fontsize=12)  # 调整X轴刻度字体大小
plt.yticks(fontsize=12)  # 调整Y轴刻度字体大小
plt.savefig(r'output/xgb_confusion.svg')
# plt.show()

y_test_pred = best_RF.predict(X_test)
# 生成混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
# cm_percentage = cm / cm.sum(axis=1, keepdims=True) 
# 可视化混淆矩阵
plt.rc('font', family = 'Times New Roman', weight='bold')
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, square=True,
            xticklabels=['unstroke', 'stroke'], yticklabels=['unstroke', 'stroke']) 
# plt.title('Confusion Matrix for RF', fontsize=15)
plt.xlabel('Predicted label', fontsize=12, fontweight = 'bold')
plt.ylabel('True label', fontsize=12)
plt.xticks(fontsize=12)  # 调整X轴刻度字体大小
plt.yticks(fontsize=12)  # 调整Y轴刻度字体大小
plt.savefig(r'output/rf_confusion.svg')
# plt.show()