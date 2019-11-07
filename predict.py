from model import MyModel
import tensorflow as tf
import pandas as pd


class Predictor:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    def predict(self, to_pre, test_path):
        # mlp模型预测
        model = MyModel.my_mlp()
        model.load_weights(self.weights_path)
        pred_labels = model.predict(to_pre)

        # 处理结果成kaggle接受的数据格式
        pred_labels = tf.argmax(pred_labels, axis=1)
        test = pd.read_csv(test_path)
        test.drop(test.columns[1:], axis=1, inplace=True)
        test.insert(1, 'Survived', pred_labels)

        return test
