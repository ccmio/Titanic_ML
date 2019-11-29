from model import MyModel
import tensorflow as tf
import pandas as pd
import numpy as np
from mlp_train import Trainer


reTrain = True


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, data, test_path='./data/test.csv', checkpoint_save_path='./data/saved_model.h5'):
        if self.model == 'mlp':
            # mlp超参数
            batch_size = 128
            epochs = 1000
            init_lr = 0.2
            '''
            参数提取/备用
            file = open('./data/pre_weights.txt', 'w')
            for v in model.trainable_weights:
                file.write(str(v.name) + '\n')
                file.write(str(v.shape) + '\n')
                file.write(str(v.numpy()) + '\n')
            file.close()
            '''
            to_pre = data[891:].copy()
            to_pre.drop(['Survived'], axis=1, inplace=True)

            if reTrain:
                trainer = Trainer(data, batch_size=batch_size, epochs=epochs, init_lr=init_lr)
                model = trainer.train()
            else:
                model = MyModel.my_mlp()
                model.load_weights(checkpoint_save_path)
            to_pre = tf.convert_to_tensor(to_pre.values, dtype=tf.float64)
            pred_labels = model.predict(to_pre)
            pred_labels = tf.argmax(pred_labels, axis=1)

        elif self.model == 'bayes':
            pred_labels = np.array(MyModel.my_bayes(data))

        elif self.model == 'ranforest':
            pred_labels = np.array(MyModel.my_ranforest(data), dtype=np.int)

        # 处理结果成kaggle接受的数据格式
        test = pd.read_csv(test_path)
        test.drop(test.columns[1:], axis=1, inplace=True)
        test.insert(1, 'Survived', pred_labels)

        return test

