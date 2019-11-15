from pretreat import Dealer
from train import Trainer
from predict import Predictor
import datetime


def main():
    # 权重文件 训练文件 预测文件
    checkpoint_save_path = './data/saved_model.tf'
    train_path = './data/train.csv'
    test_path = './data/test.csv'
    result_path = './data/gender_submission.csv'

    # 超参数
    r = 1
    batch_size = 32
    epochs = 1000
    init_lr = 0.2

    # 数据预处理
    data = Dealer(train_path, test_path)
    train_data, to_pred = data.load_clean()

    # 训练
    trainer = Trainer(train_data=train_data, batch_size=batch_size, epochs=epochs, init_lr=init_lr, r=r)
    trainer.train(checkpoint_save_path)

    # # 预测
    predictor = Predictor(checkpoint_save_path)
    pred = predictor.predict(to_pred, test_path)
    pred.to_csv(result_path, index=False)


if __name__ == '__main__':
    main()
