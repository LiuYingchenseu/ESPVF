# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
import sys
import io
import os

import json
import numpy as np
import joblib
import pandas as pd


# 设置标准输出和错误输出为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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


if len(sys.argv) > 1:
    json_str = sys.argv[1]
    data = json.loads(json_str)
    print(f"接收 JSON 数据：{data}")
    # 准备数据格式
    # 准备输入数据，将数据转换为二维数组形式
    input_data = pd.DataFrame([[
        data["age"], data["gender"], data["left_arm"], data["right_arm"],
        data["face"], data["directivity"], data["trajectory"],
        data["left_leg"], data["right_leg"], data["hypertension"]
    ]], columns=["age", "gender", "left_arm", "right_arm", "face",
                 "directivity", "trajectory", "left_leg", "right_leg", "hypertension"])
    print(f"输入数据为：{input_data}")

    model = joblib.load(r"D:\研究生文件记录\小论文资料汇总\实验代码\stroke_project\stroke_model.pkl")
    prediction = model.predict(input_data)

    print(prediction[0])
else:
    print("No input data received. ")