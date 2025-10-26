# -*- coding:UTF-8 -*-
import pandas as pd
from sklearn.model_selection import KFold
import wandb
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..//")
from data.load_data import DataGenerator
from models.model import create_multi_input_model
from utils.accuracy_loss import accuracy_loss
from utils.accuracy_loss import log_metrics_and_save_model


def model_fold_train(label_path, full_dir, mouth_dir, eye_dir, input_shape, sequence_length, epochs):
#     初始化wandb
    wandb.init(project="facial", name = "cnn_lstm_hunhe")
    
    data = pd.read_excel(label_path, engine='openpyxl')
    data['patient_id'] = data['ID'].apply(lambda x: x.split('_')[0])

    # 每个患者ID收集
    patients = data['patient_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold,(train_Index, val_Index) in enumerate(kf.split(patients), 1):
        train_patients = patients[train_Index]
        val_patients = patients[val_Index]
        train_data = data[data['patient_id'].isin(train_patients)]
        val_data = data[data['patient_id'].isin(val_patients)]

        train_labels = dict(zip(train_data['ID'], train_data['label']))
        val_labels = dict(zip(val_data['ID'], val_data['label']))
        label_map = {'normal': 0, 'palsy': 1}

        train_labels_mapped = {key: label_map[value] for key, value in train_labels.items()}
        val_labels_mapped = {key: label_map[value] for key, value in val_labels.items()}  

        print(f"Training fold {fold}")
        train_generator = DataGenerator(full_dir, mouth_dir, eye_dir, train_labels_mapped,  batch_size=1, sequence_length=16, target_size=(64,64))
        val_generator = DataGenerator(full_dir, mouth_dir, eye_dir, val_labels_mapped,  batch_size=1, sequence_length=16, target_size=(64,64))

        model = create_multi_input_model(input_shape, sequence_length)

        history = model.fit(train_generator, validation_data = val_generator, epochs=epochs, verbose=1)

        # 绘制并直接上传accuracy-loss曲线到wandb
        accuracy_loss(epochs, history, fold)

        log_metrics_and_save_model(model, val_generator, val_labels_mapped, fold)

