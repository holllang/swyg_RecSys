import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from dataloader import DataLoader
import argparse


def vectorize_sequences(sequences, dimension=40):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/example.json')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    # 데이터 전처리
    dl = DataLoader(args.data_path)
    num_per_question = [2,3,4,3,2,3,3,2,3,2,3,2,2,2,2,2]
    shape_X = sum(num_per_question)
    dl.setBias(num_per_question)
    
    X_labels = np.array([i for i in range(dl.getLen())])
    num2hobby = dl.getNum2Hobby()
    answers_with_bias = dl.getDatasetWithBias()

    X_train = vectorize_sequences(answers_with_bias)
    one_hot_train_labels = to_categorical(X_labels)

    # 새 모델로 시작
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(shape_X,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(dl.getCount(), activation='softmax'))

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train
    history = model.fit(X_train,
                    one_hot_train_labels,
                    epochs=args.epoch,
                    batch_size=args.batch_size
                    )

    # 모델 저장
    model.save('./model_saved')