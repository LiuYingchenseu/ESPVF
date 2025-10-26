# -*- coding:UTF-8 -*-
import os
import numpy as np
from utils.get_sort_key_ import get_sort_key_frame
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

'''
 数据生成器——生成可以直接传入网络的数据

    full_path:      全脸数据的地址
    mouth_path:     嘴部数据的地址
    eye_path:       眼部数据的地址

    label:          传入的是ID与对应的标签 在交叉验证分组前会对数据按照ID进行分组
    batch_size:     每次喂入模型的批次大小
    sequence_length:一批次喂入16帧图片
    target_size:    将帧图像resize成目标大小    
 
'''


nested_labels = {}
class DataGenerator(Sequence):
    def __init__(self, full_path, mouth_path, eye_path, label, batch_size=1, sequence_length=16, target_size=(64,64)):
     
     self.full_path = full_path
     self.mouth_path = mouth_path
     self.eye_path = eye_path
     self.labels = label
     self.batch_size = batch_size
     self.sequence_length = sequence_length
     self.target_size = target_size
     

     for key, value in self.labels.items():
        patient_id, data_id = key.split('_')
        patient_key = int(patient_id)
        data_key = int(data_id)

        if patient_key not in nested_labels:
            nested_labels[patient_key] = {}

        nested_labels[patient_key][data_key] = value

    # 患者ID
     self.indices = list(self.labels.keys())
    # 初始化时打乱索引
     self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indices)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __data_generation(self, indices):
        batch_full = []
        batch_mouth = []
        batch_eye = []
        batch_labels = []
        # print(indices)
        for patient_id in indices:
            
            patient_full = []
            patient_mouth = []
            patient_eye = []
          
            # patient_id = self.patients[i]
            patient_key, data_key = patient_id.split('_')

            patient_full_dir = os.path.join(self.full_path, patient_id)
            patient_mouth_dir = os.path.join(self.mouth_path, patient_id)
            patient_eye_dir = os.path.join(self.eye_path,patient_id)
        
            # 选择每个患者文件夹中的前self.sequence_length个文件名
            full_images = sorted(os.listdir(patient_full_dir), key=get_sort_key_frame)[:self.sequence_length]
            mouth_images = sorted(os.listdir(patient_mouth_dir), key=get_sort_key_frame)[:self.sequence_length]
            eye_images = sorted(os.listdir(patient_eye_dir), key=get_sort_key_frame)[:self.sequence_length]
            for j in range(self.sequence_length):
                full_img = load_img(os.path.join(patient_full_dir, full_images[j]), target_size = self.target_size)
                mouth_img = load_img(os.path.join(patient_mouth_dir, mouth_images[j]), target_size= (32,32))
                eye_img = load_img(os.path.join(patient_eye_dir, eye_images[j]), target_size= (32,32))

                full_img = img_to_array(full_img) / 255.0
                mouth_img = img_to_array(mouth_img) / 255.0
                eye_img = img_to_array(eye_img) / 255.0

                patient_full.append(full_img)
                patient_mouth.append(mouth_img)
                patient_eye.append(eye_img)

                
            batch_full.append(patient_full)
            batch_mouth.append(patient_mouth)
            batch_eye.append(patient_eye)
            batch_labels.append(nested_labels[int(patient_key)][int(data_key)]) # 添加标签
        # print("生成批次数据的形状:", np.array(batch_full).shape)
        # print("生成批次标签的形状:", np.array(batch_labels).shape)
            
        return [np.array(batch_full), np.array(batch_mouth), np.array(batch_eye)], np.array(batch_labels)