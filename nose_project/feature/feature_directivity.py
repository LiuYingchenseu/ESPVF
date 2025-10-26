'''
    1. 数据归一化
    2. 分别计算四种特征值
    3. 小提琴图观察对于类别的区分度

'''

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 定义文件夹路径
folder_path = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\混合数据\nose\nontrajectory"
# output_folder_path = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\dataset_normal\directivity_health"

# # 如果输出文件夹不存在，则创建
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

distances = []
filenames = []
# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # 检查是否为 CSV 文件
        file_path = os.path.join(folder_path, file_name)
        filenames.append(file_name.split('.')[0])
        df = pd.read_csv(file_path)

        x = df['index_tip_x'].values
        y = df['index_tip_y'].values
        
        nose_tip_x = df['nose_tip_x'].dropna().iloc[0]
        nose_tip_y = df['nose_tip_y'].dropna().iloc[0]

        index_tip_x = x[-1]
        index_tip_y = y[-1]

        distance = np.sqrt((index_tip_x - nose_tip_x) ** 2 + (index_tip_y - nose_tip_y) ** 2)

        print(nose_tip_x)
        print(nose_tip_y)

        print(index_tip_x)
        print(index_tip_y)

        print(f"鼻尖与指尖的欧式距离为：{distance}")
        distances.append(distance)

# labels = [0] * len(distances)

nose_pd = pd.read_excel(r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\label_hunhe.xlsx" )
pd.DataFrame(nose_pd)
distances_df = pd.DataFrame({
    'file_id': filenames,
    'distance': distances
})

print(distances_df)
merged_df = pd.concat([ distances_df, nose_pd], axis=1, ignore_index=False)

# 打印合并后的 DataFrame
print(nose_pd)
print(merged_df)
# distances['label'] = labels
output_directivity_health = r'D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\label_hunhe.csv'
merged_df.to_csv(output_directivity_health)
