import wandb
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                            , roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def accuracy_loss(epochs, history, fold):
    plt.figure(figsize=(15, 12))
    plt.rc('font', family = 'Times New Roman', weight='bold', size=13)
    plt.subplot(2,1,1)
    plt.plot(range(0, epochs), history.history['accuracy'])
    plt.plot(range(0, epochs), history.history['val_accuracy'])
    plt.title(f'model accuracy_loss - fold {fold}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left') 
    
    plt.subplot(2,1,2)
    plt.plot(range(0, epochs), history.history['loss'])
    plt.plot(range(0, epochs), history.history['val_loss'])
    plt.ylabel('model loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left') 
    plt.savefig(f"D:/研究生文件记录/小论文资料汇总/实验代码/face_project/ouput/accuracy_loss_plot_fold_{fold}.png")

    plt.tight_layout()
    wandb.log({f"accuracy_loss_plot_fold_{fold}": wandb.Image(plt)})
    plt.close()

def log_metrics_and_save_model(model, val_generator, val_labels_mapped, fold):
    # 验证集预测结果
    y_true = []
    y_pred = []
    y_pred_prob = []
    for i in range(len(val_generator)):
        X, y = val_generator[i]
        
        val_predictions = model.predict(X)
        y_pred_prob.extend(val_predictions.flatten()) # 保存概率
        # val_predictions = (val_predictions > 0.6).astype(int)
        y_true.extend(y)
        y_pred.extend(np.round(val_predictions).flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)  
    y_pred_prob = np.array(y_pred_prob)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(cm)

    # 混淆矩阵可视化并直接上传到wandb
    plt.rc('font', family = 'Times New Roman', weight='bold', size=13)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(f"D:/研究生文件记录/小论文资料汇总/实验代码/face_project/ouput/confusion_matrix_fold_{fold}.png")   
    plt.tight_layout()
    wandb.log({f"confusion_matrix_fold_{fold}":wandb.Image(plt)})
    plt.close()

    # 计算其他性能指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    # 记录性能指标到wandb
    wandb.log({
        f"fold_{fold}_accuracy": accuracy,
        f"fold_{fold}_precision": precision,
        f"fold_{fold}_recall": recall,
        f"fold_{fold}_f1": f1,
        f"fold_{fold}_roc_auc": roc_auc
    })

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresholds = thresholds[optimal_idx]
    print(f'Optimal threshold: {optimal_thresholds}')
    # 绘制 ROC 曲线
    plt.rc('font', family = 'Times New Roman', weight='bold', size=13)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Fold {fold}')
    plt.legend(loc="lower right")
    plt.savefig(f"D:/研究生文件记录/小论文资料汇总/实验代码/face_project/ouput/roc_curve_fold_{fold}.png")
    wandb.log({f"roc_curve_fold_{fold}": wandb.Image(plt)})
    plt.close()
    # 计算并绘制 PR 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold}')
    plt.legend(loc="lower left")
    plt.savefig(f"D:/研究生文件记录/小论文资料汇总/实验代码/face_project/ouput/pr_curve_fold_{fold}.png")
    wandb.log({f"pr_curve_fold_{fold}": wandb.Image(plt)})
    plt.close()

    # 保存模型到 wandb
    model_path = f"D:/研究生文件记录/小论文资料汇总/实验代码/face_project/ouput/model_fold_{fold}.h5"
    model.save(model_path)
    wandb.save(f"model_fold_{fold}.h5")