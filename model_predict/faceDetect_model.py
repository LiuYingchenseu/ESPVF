# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
import os
import numpy as np
import dlib
import cv2
from imutils import face_utils
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import numpy as np
import cv2
import dlib
from imutils import face_utils
import shutil

import sys
import io

# 设置标准输出和错误输出为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Starting Py Face Detection...")

class FaceDataCreator:
    def __init__(self, inputPath, outputPath, mouth_outputpath, eye_outputpath, predictor_path):
        self.input_path = inputPath
        self.output_path = self.create_dir(outputPath)
        self.mouth_path = self.create_dir(mouth_outputpath)
        self.eye_path = self.create_dir(eye_outputpath)
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    @staticmethod
    def create_dir(base_path):
        os.makedirs(base_path, exist_ok=True)
        return base_path

    def __face_extract(self):
        """Extracts face frames and samples 16 frames based on lip distance."""
        video_capture = cv2.VideoCapture(self.input_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 16:
            print(f"{total_frames},采集帧数不足16帧，重新录制！")
            video_capture.release()
            return

        frame_faces = {}
        lip_distances = []
        frame_numbers = []

        frame_count = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)

            if faces:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                lip_distance = self.calculate_lip_distance(landmarks)
                lip_distances.append(lip_distance)
                frame_numbers.append(frame_count)
                frame_faces[frame_count] = frame[face.top():face.bottom(), face.left():face.right()]

        video_capture.release()

        # Sample 16 frames based on lip distances
        self.sample_frames(frame_faces, lip_distances, frame_numbers)

    def calculate_lip_distance(self, landmarks):
        """Calculates average lip distance based on selected landmarks."""
        points = [landmarks.part(i) for i in [50, 51, 53, 54, 56, 57, 59, 60]]
        distances = [
            np.linalg.norm(np.array([points[i].x, points[i].y]) - np.array([points[i + 4].x, points[i + 4].y]))
            for i in range(4)
        ]
        return np.mean(distances)

    def sample_frames(self, frame_faces, lip_distances, frame_numbers):
        """Samples 16 frames: 3 for mouth closure start, 10 for mouth opening, 3 for mouth closure end."""
        lip_distances = np.array(lip_distances)
        min_dist, max_dist = lip_distances.min(), lip_distances.max()
        threshold_open = min_dist + (max_dist - min_dist) * 0.6
        threshold_close = min_dist + (max_dist - min_dist) * 0.3

        # Define regions for mouth closure and opening
        opening_region = [i for i, d in enumerate(lip_distances) if d >= threshold_open]
        closing_region = [i for i, d in enumerate(lip_distances) if d <= threshold_close]

        if not (opening_region and closing_region):
            print("未检测到完整的闭合-张开-闭合过程")
            return

        regions = [
            ("mouth_closure_start", closing_region[0], opening_region[0], 3),
            ("mouth_opening", opening_region[0], opening_region[-1], 10),
            ("mouth_closure_end", opening_region[-1], closing_region[-1], 3)
        ]

        sample_index = 1
        for region_name, start, end, count in regions:
            selected_frames = np.linspace(start, end, num=count, dtype=int)
            for idx in selected_frames:
                frame_path = os.path.join(self.output_path, f'sampled_frame_{sample_index}.jpg')
                cv2.imwrite(frame_path, frame_faces[frame_numbers[idx]])
                sample_index += 1

    def __mouth_eye_cut(self):
        """Cuts and saves mouth and eye regions from sampled frames."""
        for frame_file in os.listdir(self.output_path):
            frame_path = os.path.join(self.output_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            faces = self.face_detector(frame)
            if not faces:
                continue

            face = faces[0]
            landmarks = self.predictor(frame, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Cut mouth region
            mouth_frame = self.crop_region(frame, landmarks[48:68], padding=20)
            mouth_output = os.path.join(self.mouth_path, frame_file)
            cv2.imwrite(mouth_output, mouth_frame)

            # Cut eye region
            eye_frame = self.crop_region(frame, landmarks[36:48], padding=20)
            eye_output = os.path.join(self.eye_path, frame_file)
            cv2.imwrite(eye_output, eye_frame)

        # 检查并填补 mouth 和 eye 路径中的缺失帧
        self.ensure_sequence(self.mouth_path, sequence_length=16)
        self.ensure_sequence(self.eye_path, sequence_length=16)

    def crop_region(self, frame, points, padding=10):
        """Crops a region around given points with optional padding."""
        x_min, y_min = np.min(points, axis=0) - padding
        x_max, y_max = np.max(points, axis=0) + padding
        return frame[y_min:y_max, x_min:x_max]

    def ensure_sequence(self, path, sequence_length=16):
        """Checks and fills missing frames in the specified directory."""
        # 获取所有图像文件并按序号排序
        images = sorted([img for img in os.listdir(path) if img.startswith("sampled_frame_")],
                        key=lambda x: int(re.search(r"(\d+)", x).group()))

        # 检查编号缺失的情况
        existing_numbers = [int(re.search(r"(\d+)", img).group()) for img in images]
        missing_numbers = [i for i in range(1, sequence_length + 1) if i not in existing_numbers]

        # 对于缺失的编号，找到最近的邻近图像进行填补
        for missing in missing_numbers:
            nearest = min(existing_numbers, key=lambda x: abs(x - missing))
            source_image_path = os.path.join(path, f"sampled_frame_{nearest}.jpg")
            target_image_path = os.path.join(path, f"sampled_frame_{missing}.jpg")

            # 复制图像
            shutil.copy(source_image_path, target_image_path)
            print(f"Filled missing frame {missing} with frame {nearest}")

    def process_video(self):
        """Main interface to process video and save mouth and eye regions."""
        self.__face_extract()
        self.__mouth_eye_cut()
        print("视频处理完成，所有采样帧已保存。")



class PredictionDataGenerator(Sequence):
    def __init__(self, full_path, mouth_path, eye_path, patient_id, sequence_length=16, target_size=(64, 64)):
        """
        预测数据生成器，加载特定患者的图像数据
        :param full_path: 全脸图像目录
        :param mouth_path: 嘴部图像目录
        :param eye_path: 眼部图像目录
        :param patient_id: 要预测的患者ID
        :param sequence_length: 要加载的帧数
        :param target_size: 图像的目标大小
        """
        self.full_path = full_path
        self.mouth_path = mouth_path
        self.eye_path = eye_path
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.indices = [patient_id]  # 仅包含一个患者ID

    def __len__(self):
        # 仅包含一个患者ID，因此长度为1
        return 1
    def __get_sort_key_frame(self, filename):
        match = re.match(r'sampled_frame_(\d+)\.jpg', filename)
        if match:
            return int(match.group(1))
        return float('inf')  # 如果未匹配到数字，将其放在最后

    def __getitem__(self, index):
        # 加载图像数据
        patient_full, patient_mouth, patient_eye = [], [], []

        # 获取患者目录中的图像文件并排序
        full_images = sorted(os.listdir(self.full_path), key=self.__get_sort_key_frame)[:self.sequence_length]
        mouth_images = sorted(os.listdir(self.mouth_path), key=self.__get_sort_key_frame)[:self.sequence_length]
        eye_images = sorted(os.listdir(self.eye_path), key=self.__get_sort_key_frame)[:self.sequence_length]

        # 加载和预处理图像数据
        for i in range(self.sequence_length):
            full_img = load_img(os.path.join(self.full_path, full_images[i]), target_size=self.target_size)
            mouth_img = load_img(os.path.join(self.mouth_path, mouth_images[i]), target_size=(32, 32))
            eye_img = load_img(os.path.join(self.eye_path, eye_images[i]), target_size=(32, 32))

            patient_full.append(img_to_array(full_img) / 255.0)
            patient_mouth.append(img_to_array(mouth_img) / 255.0)
            patient_eye.append(img_to_array(eye_img) / 255.0)

        # 返回单个患者的数据
        return [np.array([patient_full]), np.array([patient_mouth]), np.array([patient_eye])]



# if len(sys.argv) > 1:

#     input_path = sys.argv[1]
#     if os.path.exists(input_path):
#         print(f"文件存在：{input_path}")
#     else:
#         print(f"文件路径无效：{input_path}")


input_path = r"D:\data\2024.07.10\02\2.show_teeth\2024.07.10.09.04.48.avi"
patient_id = os.path.splitext(os.path.basename(input_path))[0]
output_path = r'D:\data_myq\test_predict\full'
mouth_output_path = r'D:\data_myq\test_predict\mouth'
eye_output_path = r'D:\data_myq\test_predict\eye'
full_path = os.path.join(output_path, patient_id)
mouth_path = os.path.join(mouth_output_path, patient_id)
eye_path = os.path.join(eye_output_path, patient_id)
predictor_path = r"D:\data_myq\test_predict\shape_predictor_68_face_landmarks.dat"

face_data = FaceDataCreator(input_path, full_path, mouth_path, eye_path, predictor_path)
face_data.process_video()


predict_generator = PredictionDataGenerator(full_path, mouth_path, eye_path, patient_id, sequence_length=16, target_size=(64, 64))

# 加载训练好的模型
model = load_model(r"D:\研究生文件记录\小论文资料汇总\实验代码\face_project\ouput\2412121627_show\model_fold_1.h5")

# 进行预测
input_data = predict_generator[0]  # 因为只有一个患者，所以使用索引0获取数据
prediction = model.predict(input_data)
print(prediction)

threshold = 0.96
predicted_class = 1 if prediction >= threshold else 0

# # 映射到标签
# label_map = {0: 'normal', 1: 'palsy'}
# predicted_label = label_map[predicted_class]

print("Prediction for patient:", patient_id)
# print("Predicted Class (0=normal, 1=palsy):", predicted_class)
print("Predicted Probability:", prediction[0])
print(predicted_class)
# else:
#     print("No FacePath")
