import tf2onnx
import tensorflow as tf

model_h5_path = r"D:\研究生文件记录\小论文资料汇总\实验代码\face_project\ouput\2411062025\model_H5\model_fold_3.h5"
model_onnx_path  = r"D:\研究生文件记录\小论文资料汇总\实验代码\face_project\model_onnx.onnx"

model_h5 = tf.keras.models.load_model(model_h5_path)

spec = (tf.TensorSpec((None, 16, 64, 64, 3), tf.float32),
        tf.TensorSpec((None, 16, 32, 32, 3), tf.float32),
        tf.TensorSpec((None, 16, 32, 32, 3), tf.float32))

model_onnx, _ = tf2onnx.convert.from_keras(model_h5, input_signature=spec, opset=13)

with open(model_onnx_path, "wb") as f:
    f.write(model_onnx.SerializeToString())

print(f"ONNX 模型已保存至{model_onnx_path}")