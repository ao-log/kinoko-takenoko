import os
import sys
import glob
import argparse
import pickle
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

IMAGE_PARENT_DIR = '/workspace/image/kakou'
IMAGE_SIZE = 28

def images(category):
    return glob.glob(os.path.join(\
        '%s/%s' % (IMAGE_PARENT_DIR, category), '*.jpg'))

def category_list(comma_separatted_string):
    categories = comma_separatted_string.split(',')
    if len(categories) == 1:
        print('Error: カテゴリは二つ以上指定してください。')
        sys.exit(1)
    return categories

def to_nparray(image):
    return img_to_array(\
        load_img(image, grayscale=True, target_size=(IMAGE_SIZE, IMAGE_SIZE)))

def load_data(categories, filter = 20):
    x = []
    y = []

    for category, i in zip(categories, range(len(categories))):
        for image in images(category):
            x.append(to_nparray(image))
            y.append(i)

    x = np.array(x)
    x = np.clip(255 - x - filter, 0, 255)
    x = x.reshape(x.shape[0], IMAGE_SIZE * IMAGE_SIZE)
    x /= 255

    y = keras.utils.to_categorical(y, len(categories))

    x, y = shuffle(x, y, random_state=0)
    threshold = int(x.shape[0] * 0.8)

    return (x[:threshold] ,y[:threshold]), (x[threshold:], y[threshold:])

def define_model(categories):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(categories), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    return model

def main():
    parser = argparse.ArgumentParser(description='機械学習を行うスクリプト')
    parser.add_argument('--categories', '-c', default='default', required=True,
        help='分類対象の画像ファイルを格納しているディレクトリ名。\
            kakou/xxx の xxx の部分をコンマ区切りで指定してください。\
            例: kinoko,takenoko')
    parser.add_argument('--batch_size', '-b', default=20,
        help='バッチサイズ')
    parser.add_argument('--epochs', '-e', default=10,
        help='エポック数')
    args = parser.parse_args()

    categories = category_list(args.categories)

    (x_train, y_train), (x_test, y_test) = load_data(categories)

    callbacks = [keras.callbacks.TensorBoard(log_dir="/workspace/__tmp/", histogram_freq=1)]

    model = define_model(categories)
    #model.summary()

    history = model.fit(x_train, y_train,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        verbose=1,
        callbacks=callbacks,
        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('/workspace/models/ml.model.h5')

    with open('/workspace/__tmp/ml.history.pickle', 'wb') as f:
                pickle.dump(history.history, f)

if __name__ == '__main__':
      main()

