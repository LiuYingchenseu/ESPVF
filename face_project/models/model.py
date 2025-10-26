# -*- coding:UTF-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout, TimeDistributed, Concatenate, Multiply, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# def attention_block(inputs, name):
#     attention_probs = Dense(inputs.shape[-1], activation='softmax', name=name+'_attention')(inputs)
#     attention_mul = Multiply(name=name+'_attention_mul')([inputs, attention_probs])
#     return attention_mul

# class CBAMLayer(Layer):
#     def __init__(self, reduction_ratio=8, **kwargs):
#         super(CBAMLayer, self).__init__(**kwargs)
#         self.reduction_ratio = reduction_ratio

#     def build(self, input_shape):
#         # Channel Attention Module
#         channel = input_shape[-1]
#         self.shared_layer = Dense(channel // self.reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
#         self.shared_layer2 = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

#         # Spatial Attention Module
#         self.conv = Conv2D(1, (7, 7), strides=(1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

#     def call(self, inputs):
#         # Channel Attention
#         print(f"CBAMLayer received input shape: {inputs.shape}")
#         avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
#         max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        
#         avg_pool = self.shared_layer(avg_pool)
#         avg_pool = self.shared_layer2(avg_pool)

#         max_pool = self.shared_layer(max_pool)
#         max_pool = self.shared_layer2(max_pool)

#         cbam_feature = Add()([avg_pool, max_pool])
#         cbam_feature = Activation('sigmoid')(cbam_feature)
#         cbam_feature = Multiply()([inputs, cbam_feature])

#         # Spatial Attention
#         avg_pool = tf.reduce_mean(cbam_feature, axis=-1, keepdims=True)
#         max_pool = tf.reduce_max(cbam_feature, axis=-1, keepdims=True)
#         concat = Concatenate(axis=-1)([avg_pool, max_pool])
#         cbam_feature = self.conv(concat)

#         output = Multiply()([inputs, cbam_feature])
#         return output

    # def compute_output_shape(self, input_shape):
    #     print(f"Input shape: {input_shape}")
    #     # input_shape is (batch_size, time_steps, height, width, channels)
    #     # batch_size, time_steps, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        
    #     # The output shape should match the input shape
    #     return input_shape
    # def get_config(self):
    #     config = super(CBAMLayer, self).get_config()
    #     config.update({
    #         'reduction_ratio':self.reduction_ratio,
    #     })
    #     return config

# # 自定义 HW注意力层
# class HWAttentionLayer(Layer):
#     def __init__(self, reduction_ratio=8, **kwargs):
#         super(HWAttentionLayer, self).__init__(**kwargs)
#         self.reduction_ratio = reduction_ratio

#     def build(self, input_shape):
#         channel = input_shape[-1]
#         self.shared_dense = Dense(channel // self.reduction_ratio, activation='relu')
#         self.final_dense = Dense(channel, activation='sigmoid')

#     def call(self, inputs):
#         avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
#         max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
        
#         avg_out = self.shared_dense(avg_pool)
#         max_out = self.shared_dense(max_pool)
        
#         attention = self.final_dense(avg_out + max_out)
#         return Multiply()([inputs, attention])

#     def compute_output_shape(self, input_shape):
#         # 返回输入的形状，因为HWAttentionLayer不会改变张量的形状
#         return input_shape

def create_cnn_lstm_branch(input_shape, sequence_length, branch_name):
    inputs = Input(shape=(sequence_length,) + input_shape, name=branch_name + '_input')
    print('Shape after inputs:', inputs.shape)   
    x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)),
                         name=branch_name + '_conv1')(inputs)
    print('Shape after Conv2D:', x.shape)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name=branch_name + '_pool1')(x)
    print('Shape after MaxPooling2D:', x.shape)

    # 使用自定义 CBAM 层
    # cbam_layer = CBAMLayer(name=branch_name + '_cbam')
    # x = TimeDistributed(cbam_layer, name=branch_name + '_cbam')(x)

    x = TimeDistributed(Flatten(), name=branch_name + '_flatten')(x)
    x = LSTM(32, return_sequences=True, name=branch_name + '_lstm')(x)
    x = Dropout(0.3, name=branch_name + '_lstm_dropout')(x) 
    # x = TimeDistributed(HWAttentionLayer(), name=branch_name + '_hw_attention')(x)
    x = GlobalAveragePooling1D()(x)

    # x = attention_block(x, branch_name)
    # x = Dropout(0.5, name=branch_name + '_dropout')(x)

    return inputs, x

def create_multi_input_model(input_shape, sequence_length):
    # 全脸通道
    full_inputs, full_branch = create_cnn_lstm_branch(input_shape, sequence_length, 'full_face')
    # full_branch = cbam_block(full_branch, name='full_face_cbam')
    # 嘴巴通道
    mouth_inputs, mouth_branch = create_cnn_lstm_branch((32, 32, 3), sequence_length, 'mouth')
    # mouth_branch = cbam_block(mouth_branch, name='mouth_cbam')
    # 眼睛通道
    eye_inputs, eye_branch = create_cnn_lstm_branch((32, 32, 3), sequence_length, 'eyes')
    # eye_branch = cbam_block(eye_branch, name='eyes_cbam')
    # 合并所有分支的特征
    merged = Concatenate(name='concat')([full_branch, mouth_branch, eye_branch])

    # 使用全局平均池化层处理时间维度，减少到标量输出
    # x = GlobalAveragePooling1D(name='gap1d')(merged)

    # 全连接层和输出层
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01), name='dense1')(merged)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    # 创建模型
    model = Model(inputs=[full_inputs, mouth_inputs, eye_inputs], outputs=outputs)

    # 调整学习率
    learning_rate = 0.0001
    optimizer = Adam(learning_rate = learning_rate, clipnorm = 1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

