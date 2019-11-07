from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation


class MyModel:
    def __init__(self):
        pass

    @staticmethod
    def my_mlp():
        model = Sequential()
        model.add(Dense(input_shape=(7,), units=16))  # 不写input_shape会报无法载入weights的错！
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(units=2))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        return model

    def my_ranforest(self):
        pass

    def my_bayes(self):
        pass
