from generations import *
import tensorflow as tf
import scipy.io as sio


def bit_err(y_true, y_pred):
    # 将预测值和真实值转换为0和1，基于0.5的阈值
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_bin = tf.cast(y_true > 0.5, tf.float32)

    # 计算位错误率：两者不等的平均值
    return tf.reduce_mean(tf.abs(y_pred_bin - y_true_bin))


input_bits = tf.keras.Input(shape=(payloadBits_per_OFDM,))
temp = tf.keras.layers.BatchNormalization()(input_bits)
temp = tf.keras.layers.Dense(n_hidden_1, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_2, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_3, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
out_put = tf.keras.layers.Dense(n_output, activation='sigmoid')(temp)
model = tf.keras.Model(input_bits, out_put)
model.compile(optimizer='adam', loss='mse', metrics=[bit_err])  # bit_err为评估函数，自动输入真实标签和预测标签，tensorflow功能；
model.summary()
# 定义检查点回调:
# ModelCheckpoint 回调被创建，配置为在验证位错误率（val_bit_err）最小时保存模型的权重。这是为了在训练过程中能够自动保存表现最佳的模型权重，而不是预先存储模型。
# TensorFlow 简化并行和异步计算，开发者不必直接管理线程或进程。深度学习训练过程中的并行化通常关注于计算操作（如张量运算）和数据处理
# （如数据加载和预处理）的并行化， TensorFlow 的底层实现自动管理，如 GPU / CPU 并行计算。
checkpoint = tf.keras.callbacks.ModelCheckpoint('./temp_trained_25.h5', monitor='val_bit_err',
                                                verbose=0, save_best_only=True, mode='min', save_weights_only=True)
# 训练模型；
model.fit(
    training_gen(1000, 25),
    steps_per_epoch=50,
    epochs=100,
    validation_data=validation_gen(1000, 25),
    validation_steps=1,
    callbacks=[checkpoint],
    verbose=2)

model.load_weights('./temp_trained_25.h5')
BER = []
for SNR in range(5, 30, 5):
    y = model.evaluate(
        validation_gen(1000, SNR),
        steps=1
    )
    BER.append(y[1])
    print(y)
print(BER)
BER_matlab = np.array(BER)
sio.savemat('BER_4QAM_32.mat', {'BER': BER_matlab})
