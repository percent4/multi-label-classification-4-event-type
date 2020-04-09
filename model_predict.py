# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-04-03 21:50

import json
import numpy as np
from keras.models import load_model

from att import Attention
from albert_zh.extract_feature import BertVector
load_model = load_model("event_type.h5", custom_objects={"Attention": Attention})

# 预测语句
text = "昨天18：30，陕西宁强县胡家坝镇向家沟村三组发生山体坍塌，5人被埋。当晚，3人被救出，其中1人在医院抢救无效死亡，2人在送医途中死亡。今天凌晨，另外2人被发现，已无生命迹象。"
text = text.replace("\n", "").replace("\r", "").replace("\t", "")

labels = []

bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)

# 将句子转换成向量
vec = bert_model.encode([text])["encodes"][0]
x_train = np.array([vec])

# 模型预测
predicted = load_model.predict(x_train)[0]

indices = [i for i in range(len(predicted)) if predicted[i] > 0.5]

with open("event_type.json", "r", encoding="utf-8") as g:
    movie_genres = json.loads(g.read())

print("预测语句: %s" % text)
print("预测事件类型: %s" % "|".join([movie_genres[index] for index in indices]))

