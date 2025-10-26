import sys
import json
import numpy as np
import joblib  # 或 pickle，用于加载模型
import pandas as pd


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
# 输入数据
data = {
    "age": 46,
    "gender": 0,
    "directivity": 1,
    "face": 1,
    "hypertension": 2,
    "left_arm": 2,
    "left_leg": 2,
    "right_arm": 2,
    "right_leg": 2,
    "trajectory": 2
}


# 将输入数据转换为 DataFrame
feature_names = ["age", "gender", "left_arm", "right_arm", "face", 
                 "directivity", "trajectory", "left_leg", "right_leg", "hypertension"]

input_data = pd.DataFrame([data], columns=feature_names)

model = joblib.load(r"D:\研究生文件记录\小论文资料汇总\实验代码\stroke_project\stroke_model.pkl")

prediction = model.predict(input_data)

print(prediction)