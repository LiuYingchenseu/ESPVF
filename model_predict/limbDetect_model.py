import sys
import io
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.optimize import curve_fit

# 设置标准输出和错误输出为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 初始化 mediapipe 模型
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class LimbDetector:
    def __init__(self, inputPath):
        self.input_path = inputPath
        self.frame_height = 720
        self.frame_width = 720
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __detect_keypoints(self):
        index_tip = []
        cap = cv2.VideoCapture(self.input_path)
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = self.pose.process(rgb_frame)
            hands_results = self.hands.process(rgb_frame)

            nose_tip_x, nose_tip_y = np.nan, np.nan
            finger_tip = (np.nan, np.nan)

            if pose_results.pose_landmarks:
                nose_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                nose_tip_x = int(nose_landmark.x * frame.shape[1])
                nose_tip_y = int(nose_landmark.y * frame.shape[0])

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    finger_tip = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))
                    break

            index_tip.append({
                "frame": frame_num,
                "index_tip_x": finger_tip[0],
                "index_tip_y": finger_tip[1]
            })

        cap.release()
        return pd.DataFrame(index_tip), nose_tip_x, nose_tip_y

    def __keypoint_normalization(self):
        index_tip_df, nose_tip_x, nose_tip_y = self.__detect_keypoints()
        if index_tip_df.empty:
            raise ValueError("未检测到关键点")

        # 反转 y 坐标
        index_tip_df['index_tip_y'] = self.frame_height - index_tip_df['index_tip_y']
        nose_tip_y = self.frame_height - nose_tip_y

        # 归一化
        index_tip_df['index_tip_x'] = index_tip_df['index_tip_x'] / self.frame_width
        index_tip_df['index_tip_y'] = index_tip_df['index_tip_y'] / self.frame_height
        return index_tip_df, nose_tip_x / self.frame_width, nose_tip_y / self.frame_height

    def __quadratic_func(self, x, a, b, c):
        return a * x**2 + b * x + c

    def __r_squared(self, x, y):
        try:
            params, _ = curve_fit(self.__quadratic_func, x, y)
            y_fit = self.__quadratic_func(x, *params)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        except Exception:
            return np.nan

    def __curvature_change_rate(self, x, y):
        
        x = np.array(x)
        y = np.array(y)
        dx = np.diff(x)
        dy = np.diff(y)
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        denominator = (dx[:-1]**2 + dy[:-1]**2)**1.5
        denominator[denominator == 0] = np.nan
        curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / denominator
        curvature_changes = np.abs(np.diff(curvature))
        return np.mean(curvature_changes) if len(curvature_changes) > 0 else np.nan

    

    
    def __mean_slope_abs(self, x, y):
        slopes = []
        for i in range(1, len(x)):
            if x[i] != x[i - 1]:  # 确保分母不为零
                slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
                slopes.append(np.abs(slope))
            else:
                slopes.append(np.nan)  # 如果分母为零，设为 NaN
        slopes = np.array(slopes)
        slopes = slopes[~np.isnan(slopes)]  # 去除 NaN 值
        if len(slopes) == 0:
            return np.nan  # 若所有斜率都为 NaN，返回 NaN
        return np.mean(slopes)


    def __mean_triangle_area(self, x, y):
        if len(x) < 3:
            return np.nan
        triangle_areas = [
            0.5 * abs(x[i-1]*(y[i]-y[i+1]) + x[i]*(y[i+1]-y[i-1]) + x[i+1]*(y[i-1]-y[i]))
            for i in range(1, len(x) - 1)
        ]
        return np.mean(triangle_areas)

    def __speed_std(self, index_tip_df):
        fps = 30
        time_intervals = np.diff(index_tip_df['frame']) / fps
        distances = np.sqrt(np.diff(index_tip_df['index_tip_x'])**2 + np.diff(index_tip_df['index_tip_y'])**2)
        speeds = distances / time_intervals
        return np.std(speeds)

    def get_features(self):
        index_tip_df, nose_tip_x, nose_tip_y = self.__keypoint_normalization()
        index_tip_df_clean = index_tip_df.dropna(subset=['index_tip_x', 'index_tip_y'])
        x = index_tip_df_clean['index_tip_x'].values
        y = index_tip_df_clean['index_tip_y'].values

        r_squared = self.__r_squared(x, y)
        mean_curvature_change_rate = self.__curvature_change_rate(x, y)
        mean_slope_abs = self.__mean_slope_abs(x, y)
        mean_triangle_area = self.__mean_triangle_area(x, y)
        speed_std = self.__speed_std(index_tip_df_clean)

        index_tip_x = x[-1] if len(x) > 0 else np.nan
        index_tip_y = y[-1] if len(y) > 0 else np.nan
        distance = np.sqrt((index_tip_x - nose_tip_x)**2 + (index_tip_y - nose_tip_y)**2) if not np.isnan(index_tip_x) and not np.isnan(index_tip_y) else np.nan

        features = np.array([ mean_curvature_change_rate, mean_slope_abs, mean_triangle_area]).reshape(1, -1)
        return features, distance


# 运行脚本
if len(sys.argv) > 1:
    input_path = sys.argv[1]
    if os.path.exists(input_path):
        print(f"文件存在：{input_path}")
    # input_path = r"D:\data_myq\2024.11.30\zyt\2.show_nose\a_10_12_58.avi"
        limb_detector = LimbDetector(input_path)
        features, distance = limb_detector.get_features()
        # 处理 NaN 和无效值
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        distance = 0.0 if np.isnan(distance) else distance
        print(f"features:{features}, distance:{distance}")
        # # 特征缩放并进行模型预测
        # scaler = joblib.load(r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\output\model_pkl\MCCR_MSA_MTA_scaler.pkl")
        # scaled_features = scaler.fit_transform(features)
        # print(scaled_features)
        model = joblib.load(r"D:\研究生文件记录\小论文资料汇总\实验代码\nose_project\output\showPhoto\MCCR_MSA_MTA_model.pkl")
        prediction = model.predict(features)
        prediction_label = str(prediction[0])
        print(str(prediction_label))
        threshold = 0.07107
        if distance > threshold:
            print("1")
        else:
            print("0")
    else:
        print("文件路径无效:", input_path)
else:
    print("路径输入出错！")