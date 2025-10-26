from data.load_data import DataGenerator
from models.model_training import model_fold_train
from models.model import create_multi_input_model

if __name__ == "__main__":
    
    full_path  = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\混合数据\face\full"
    eye_path   = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\混合数据\face\eye"
    mouth_path = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\混合数据\face\mouth"
    label_path = r"G:\myq_workspace\paperbackup\paper-stroke\experimen\混合数据\face\label_hunhe.xlsx"
    
    input_shape = (64,64,3)
    sequence_length = 16
    model = create_multi_input_model(input_shape, sequence_length)
    model.summary()
    epochs = 20

    model_fold_train(label_path, full_path, mouth_path, eye_path, input_shape, sequence_length, epochs)