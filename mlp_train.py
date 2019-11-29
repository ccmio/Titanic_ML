import numpy as np
from model import MyModel
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息
load_pre = False  # 是否装载本地权重文件的开关


class Trainer:
    def __init__(self, data, batch_size=32, epochs=20, init_lr=0.02):
        train = data[:891].copy()
        self.train_data = train.values
        self.batch_size = batch_size
        self.epochs = epochs  # 每轮训练epochs次
        self.init_lr = init_lr
    
    def train(self, checkpoint_save_path='./data/saved_model.h5'):

        x_train = np.array(self.train_data[:, 1:], dtype=np.float)
        y_train = np.array(self.train_data[:, :1], dtype=np.float)
        np.random.seed(100)
        np.random.shuffle(x_train)
        np.random.seed(100)
        np.random.shuffle(y_train)
        model = MyModel.my_mlp()

        '''
        lr = 0.2
        decay_steps=14,
        decay_rate=0.99,
        SGD
        batch_size = 128
        epochs = 1000
        val_acc = 0.7989
        '''
        # 指数衰减学习率
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.init_lr,
            decay_steps=14,
            decay_rate=0.99,
            staircase=True)
        optimizer = SGD(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
        
        if load_pre and os.path.exists(checkpoint_save_path):
            model.load_weights(checkpoint_save_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         monitor='loss',
                                                         save_best_only=True,
                                                         verbose=1)

        h = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_freq=1, validation_split=0.2, verbose=1, callbacks=[cp_callback])
        model.save_weights(checkpoint_save_path)

        file = open('./data/train_weights.txt', 'w')  # 参数提取
        for v in model.trainable_weights:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')
        file.close()

        acc = h.history['sparse_categorical_accuracy']
        val_acc = h.history['val_sparse_categorical_accuracy']
        loss = h.history['loss']
        val_loss = h.history['val_loss']

        plt.figure(figsize=(14, 8))
        
        plt.subplot(1, 2, 1)
        plt.plot(acc, c='r', label='Training acc')
        plt.plot(val_acc, c='b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()  # 绘制图例，默认在右上角
        
        plt.subplot(1, 2, 2)
        plt.plot(loss, c='r', label='Training loss')
        plt.plot(val_loss, c='b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        plt.show()

        return model
