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

X = df.drop('labels', axis=1)
y = df['labels']
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# STOME 类别平衡

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Class distribution before resampling: {y_train.value_counts()}")
print(f"Class distribution after resampling: {y_train_res.value_counts()}")

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
    
def drawRoc_three(name_file,roc_auc,fpr,tpr,roc_auc1,fpr1,tpr1,roc_auc2,fpr2,tpr2):
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(10) 
    plt.rc('font', family = 'Times New Roman', weight='bold')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='RF ROC curve (area = %0.4f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='b', lw=2, label='XGBoost ROC curve (area = %0.4f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='g', lw=2, label='XGBRForest ROC curve (area = %0.4f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1],lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.size'] = 12
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.xticks(np.arange(0, 1.05, 0.2))  # 设置 x 轴刻度为从 0 到 10，间隔为 1
    plt.yticks(np.arange(0, 1.05, 0.2))  # 设置 y 轴刻度为从 -1 到 1，间隔为 0.2
    plt.title('ROC Curve',fontsize=15)
    plt.grid()
    plt.legend(loc="lower right",fontsize=12)
    
    plt.savefig(name_file)
    plt.show()

best_RF = joblib.load(r"output/best_RF.pkl")
best_xgb = joblib.load(r"output/best_xgb.pkl")
xgbrforest = joblib.load(r"output/xgbrforest.pkl")

best_RF.fit(X_train_res, y_train_res)
best_xgb.fit(X_train_res, y_train_res)
# xgbrforest.fit(X_train_res, y_train_res)

metrics = joblib.load(r'output/model_metrics.pkl')
print(metrics)


