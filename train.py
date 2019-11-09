import numpy as np
from model import MyModel
import os
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息
load_pre = True  # 是否装载本地权重文件的开关


class Trainer:
    def __init__(self, train_data, batch_size=32, epochs=20, r=10, init_lr=0.02):
        self.train_data = train_data.values
        self.batch_size = batch_size
        self.epochs = epochs  # 每轮训练epochs次
        self.r = r  # 每轮保存一次参数，训练r轮
        self.init_lr = init_lr
    
    def train(self, weights_path):
        x_train = self.train_data[:, 1:]
        y_train = self.train_data[:, :1].squeeze()
        np.random.seed(100)
        np.random.shuffle(x_train)
        np.random.seed(100)
        np.random.shuffle(y_train)

        model = MyModel.my_mlp()

        acc = []
        val_acc = []
        loss = []
        val_loss = []
        '''
        lr = 0.2
        decay_steps=120,
        decay_rate=0.96,
        SGD
        round = 50
        batch_size = 32
        epochs = 20
        val_acc = 0.8156 
        '''
        # 指数衰减学习率
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.init_lr,
            decay_steps=120,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
        
        if load_pre and os.path.exists(weights_path):
            model.load_weights(weights_path)
        
        for i in range(self.r):
            print('\n\n\n====================round {}==================\n\n\n'.format(i))
            h = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_freq=1, validation_split=0.2)
            model.save_weights(weights_path)
            acc += h.history['sparse_categorical_accuracy']
            val_acc += h.history['val_sparse_categorical_accuracy']
            loss += h.history['loss']
            val_loss += h.history['val_loss']
        model.summary()

        plt.figure(figsize=(8, 8))
        
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
