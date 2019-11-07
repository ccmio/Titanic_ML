from pretreat import Dealer
from train import Trainer
from predict import Predictor


def main():
    # 权重文件 训练文件 预测文件
    weights_path = './model_weights.h5'
    train_path = './train.csv'
    test_path = './test.csv'

    # 超参数
    r = 10
    batch_size = 32
    epochs = 20
    init_lr = 0.2

    # 数据预处理
    data = Dealer(train_path, test_path)
    train_data, to_pred = data.load_clean()

    # 训练
    trainer = Trainer(train_data=train_data, batch_size=batch_size, epochs=epochs, init_lr=init_lr, r=r)
    trainer.train(weights_path)

    # 预测
    predictor = Predictor(weights_path)
    pred = predictor.predict(to_pred, test_path)
    pred.to_csv('./gender_submission.csv', index=False)


if __name__ == '__main__':
    main()
