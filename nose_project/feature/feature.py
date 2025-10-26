import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

input_dir_path = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\nose_dataset\nose_video\data_myq\unhealth"
file_list = os.listdir(input_dir_path)
# print(file_list)
csv_files = [file for file in file_list if file.endswith('.csv')]
print(csv_files)

file_ids = []
r_squareds = []
mean_slope_abses = []
mean_triangle_areas = []
speed_stds = []
_distance = []

for csv_file in csv_files:
    csv_file_path = os.path.join(input_dir_path, csv_file)
    df = pd.read_csv(csv_file_path)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['index_tip_x', 'index_tip_y'])
    x = df_clean['index_tip_x'].values
    y = df_clean['index_tip_y'].values
    # print(df_clean)

    file_name = os.path.basename(csv_file_path)
    print(file_name)
    file_id = os.path.splitext(file_name)[0]
    print(file_id)
    file_ids.append(file_id)

    # --------------------拟合优度-----------------------#
    # 定义拟合函数
    def quadratic_func(x,a,b,c):
        return a * x**2 + b * x + c

    # 使用最小二乘法进行拟合
    params, covatiance = curve_fit(quadratic_func, x, y)

    # 拟合参数
    a, b, c = params
    # 计算拟合值
    y_fit = quadratic_func(x, a, b, c)

    # 计算拟合优度
    residuals  = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # 输出拟合参数和 R-squared
    print(f'拟合参数: a = {a:.3f}, b = {b:.3f}, c = {b:.3f}')
    print(f'拟合优度: (R-squared): {r_squared}')

    r_squareds.append(r_squared)
    # # 可视化结果
    # plt.scatter(x, y, color = 'blue',label = 'index_tip')
    # plt.plot(x, y_fit, color='yellow', label='fit')
    # x_nose = df['nose_tip_x'].iloc[-1]
    # y_nose = df['nose_tip_y'].iloc[-1]
    # plt.scatter(x_nose, y_nose, color='red', label = 'nose_tip')

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
    # ------------------斜率绝对值均值----------------------#
    slopes = []
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            slope = (y[i] - y[i-1]) / (x[i] - x[i-1])
            slopes.append(slope)
    # 绝对值
    slopes_abs = np.abs(slopes)
    # 绝对值均值
    mean_slope_abs = np.mean(slopes_abs)

    print(f'指尖关键点斜率绝对值的均值：{mean_slope_abs}')
    mean_slope_abses.append(mean_slope_abs)

    # -------------------三角形相邻面积---------------------#
    if len(x) < 3:
        print("数据点不足")
    else:
        triangle_areas = []
        for i in range(1, len(x) - 1):
            x1, y1 = x[i-1], y[i-1]
            x2, y2 = x[i], y[i]
            x3, y3 = x[i+1], y[i+1]

            area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

            triangle_areas.append(area)

        total_area = np.sum(triangle_areas)
        mean_triangle_area = total_area / len(triangle_areas)


        print(f'所有相邻三角形面积的总和: {total_area}')
        print(f'指尖关键点相邻三角形面积均值：{mean_triangle_area}')
        mean_triangle_areas.append(mean_triangle_area)
    # -----------------速度标准差-------------------------- #
    fps = 30
    time_intervals = np.diff(df_clean['frame']) / fps
    print(f'时间间隔为：{time_intervals}')
    distances = np.sqrt(
        np.diff(df_clean['index_tip_x'])**2 + np.diff(df_clean['index_tip_y'])**2
    )
    # 计算每帧的速度
    speeds = distances / time_intervals

    # 计算均速 v'
    mean_speed = np.mean(speeds)

    # 计算速度标准差s
    speed_std = np.std(speeds)

    print(f"均速：{mean_speed}, 速度标准差：{speed_std}")
    speed_stds.append(speed_std)
    # ------------------鼻尖与指尖距离---------------------- #
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
    _distance.append(distance)
# --------------------------创建新的df---------------------------- #
data = {
    'ID': file_ids,
    'R-squared': r_squareds,
    'mean_slope_abs': mean_slope_abses,
    'mean_triangle_area': mean_triangle_areas,
    'speed_std': speed_stds,
    'distance': _distance
}
labels = [0] * len(file_ids)
data['label'] = labels
health_df = pd.DataFrame(data)
output_health_path = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\nose_dataset\nose_video\data_myq\unhealth\unhealth.csv"
health_df.to_csv(output_health_path, index=False)
print(health_df)