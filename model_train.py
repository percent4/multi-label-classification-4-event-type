# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-04-03 18:12

import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from att import Attention
from keras.layers import GRU, Bidirectional
from tqdm import tqdm
import matplotlib.pyplot as plt

from albert_zh.extract_feature import BertVector

with open("./data/multi-classification-train.txt", "r", encoding="utf-8") as f:
    train_content = [_.strip() for _ in f.readlines()]

with open("./data/multi-classification-test.txt", "r", encoding="utf-8") as f:
    test_content = [_.strip() for _ in f.readlines()]

# 获取训练集合、测试集的事件类型
movie_genres = []

for line in train_content+test_content:
    genres = line.split(" ", maxsplit=1)[0].split("|")
    movie_genres.append(genres)

# 利用sklearn中的MultiLabelBinarizer进行多标签编码
mlb = MultiLabelBinarizer()
mlb.fit(movie_genres)

print("一共有%d种事件类型。" % len(mlb.classes_))

with open("event_type.json", "w", encoding="utf-8") as h:
    h.write(json.dumps(mlb.classes_.tolist(), ensure_ascii=False, indent=4))

# 对训练集和测试集的数据进行多标签编码
y_train = []
y_test = []

for line in train_content:
    genres = line.split(" ", maxsplit=1)[0].split("|")
    y_train.append(mlb.transform([genres])[0])

for line in test_content:
    genres = line.split(" ", maxsplit=1)[0].split("|")
    y_test.append(mlb.transform([genres])[0])

y_train = np.array(y_train)
y_test = np.array(y_test)

print(y_train.shape)
print(y_test.shape)

# 利用ALBERT对x值（文本）进行编码
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]

x_train = []
x_test = []

process_bar = tqdm(train_content)

for ch, line in zip(process_bar, train_content):
    movie_intro = line.split(" ", maxsplit=1)[1]
    x_train.append(f(movie_intro))

process_bar = tqdm(test_content)

for ch, line in zip(process_bar, test_content):
    movie_intro = line.split(" ", maxsplit=1)[1]
    x_test.append(f(movie_intro))

x_train = np.array(x_train)
x_test = np.array(x_test)

print("end encoding")
print(x_train.shape)


# 深度学习模型
# 模型结构：ALBERT + 双向GRU + Attention + FC
inputs = Input(shape=(200, 312, ), name="input")
gru = Bidirectional(GRU(128, dropout=0.2, return_sequences=True), name="bi-gru")(inputs)
attention = Attention(32, name="attention")(gru)
num_class = len(mlb.classes_)
output = Dense(num_class, activation='sigmoid', name="dense")(attention)
model = Model(inputs, output)

# 模型可视化
# from keras.utils import plot_model
# plot_model(model, to_file='multi-label-model.png', show_shapes=True)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=10)
model.save('event_type.h5')


# 训练结果可视化
# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("loss_acc.png")
