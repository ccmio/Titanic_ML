from pretreat import Dealer
from predict import Predictor


def main():
    # 权重文件 训练文件 预测文件
    train_path = './data/train.csv'
    test_path = './data/test.csv'
    result_path = './data/gender_submission.csv'

    # 数据预处理
    data = Dealer(train_path, test_path)
    data = data.load_clean()

    # 预测模型可选：mlp, bayes, ranforest
    print('\n==================== Predicting... ====================\n')
    predictor = Predictor('mlp')
    pred_result = predictor.predict(data)
    pred_result.to_csv(result_path, index=False)
    print('================= Prediction Generated. =================\n')


if __name__ == '__main__':
    main()
