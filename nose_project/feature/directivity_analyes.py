import pandas as pd

# 加载两个 CSV 文件
file1 = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\directivity_health.csv"
file2 = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\directivity_unhealth.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 合并两个 DataFrame
df_combined = pd.concat([df1, df2], axis=0)

# df_combined.to_csv(r'D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\directivity.csv', index=False)




file3 = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_health.csv"
file4 = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_unhealth_test.csv"

df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

# 合并两个 DataFrame
df_combined = pd.concat([df3, df4], axis=0)

df_combined.to_csv(r'D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_test.csv', index=False)