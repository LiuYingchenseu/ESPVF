import os
import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit


# 定义文件夹路径
folder_path = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_unhealth"
# output_folder_path = r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\dataset_normal\directivity_health"

# # 如果输出文件夹不存在，则创建
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

file_ids = []
r_squareds = []
mean_curvature_change_rates = []
mean_slope_abses = []
mean_triangle_areas = []
speed_stds = []
distance_threshold = 0.01 
# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # 检查是否为 CSV 文件

        file_id = os.path.splitext(file_name)[0]
        print(file_id)
        file_ids.append(file_id)

        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # x = np.round(df['index_tip_x'].values, 3)
        # y = np.round(df['index_tip_y'].values, 3)
        x = df['index_tip_x'].values
        y = df['index_tip_y'].values

        # 合并相近点
        filtered_x = [x[0]]
        filtered_y = [y[0]]
        for i in range(1, len(x)):
            distance = np.sqrt((x[i] - filtered_x[-1])**2 + (y[i] - filtered_y[-1])**2)
            if distance > distance_threshold:  # 距离超过阈值时保留点
                filtered_x.append(x[i])
                filtered_y.append(y[i])

        x = np.array(filtered_x)
        y = np.array(filtered_y)

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
        # residuals  = y - y_fit
        ss_res = np.sum((y_fit - np.mean(y))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = ss_res / ss_tot

        # 输出拟合参数和 R-squared
        print(f'拟合参数: a = {a:.3f}, b = {b:.3f}, c = {b:.3f}')
        print(f'拟合优度: (R-squared): {r_squared}')

        r_squareds.append(r_squared)

        # 定义曲率变化率函数
        def curvature_change_rate(x, y):

            x = np.array(x)
            y = np.array(y)

            dx = np.diff(x)
            dy = np.diff(y)

            ddx = np.diff(dx)
            ddy = np.diff(dy)

            denominator = (dx[:-1]**2 + dy[:-1]**2)**1.5
            denominator[denominator == 0] = np.nan
            curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / denominator
    
            # 曲率变化率
            curvature_changes = np.abs(np.diff(curvature))
            return np.mean(curvature_changes) if len(curvature_changes) > 0 else np.nan
        mean_curvature_change_rate = curvature_change_rate(x, y)
        print(f"平均曲率变化率：{mean_curvature_change_rate}")
        mean_curvature_change_rates.append(mean_curvature_change_rate)

        # 定义斜率绝对值均值
        slopes = []
        for i in range(1, len(x)):
            if x[i] != x[i - 1]:
                slope = (y[i] - y[i-1]) / (x[i] - x[i-1])
                slopes.append(slope)
        # 绝对值
        slopes_abs = np.abs(slopes)
        # 绝对值均值
        mean_slope_abs = np.mean(slopes_abs)

        print(f'指尖关键点斜率绝对值的均值：{mean_slope_abs}')
        mean_slope_abses.append(mean_slope_abs)

        # 三角形面积均值
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
        # 速度标准差
        fps = 30
        time_intervals = np.diff(df['frame']) / fps
        print(f'时间间隔为：{time_intervals}')
        distances = np.sqrt(
            np.diff(df['index_tip_x'])**2 + np.diff(df['index_tip_y'])**2
        )
        # 计算每帧的速度
        speeds = distances / time_intervals

        # 计算均速 v'
        mean_speed = np.mean(speeds)

        # 计算速度标准差s
        speed_std = np.std(speeds)

        print(f"均速：{mean_speed}, 速度标准差：{speed_std}")
        speed_stds.append(speed_std)



labels = [1] * len(file_ids)

data  = {
    "file_id": file_ids, 
    "R_squared": r_squareds, 
    "mean_curvetrue_change_rate": mean_curvature_change_rates, 
    "mean_slope_abs": mean_slope_abses,
    "mean_triangle_area":mean_triangle_areas, 
    "speed_std":speed_stds,
    "label": labels
    
}

df = pd.DataFrame(data)
print(df)
output_trajectory_unhealth = r'D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\dataset\trajectory_unhealth_test_std.csv'
df.to_csv(output_trajectory_unhealth)