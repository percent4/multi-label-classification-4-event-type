# -*- coding: utf-8 -*-
# @Time : 2020/12/23 15:28
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Yangpu, Shanghai
# 模型评估脚本,利用hamming_loss作为多标签分类的评估指标，该值越小模型效果越好
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import hamming_loss, classification_report
from att import Attention
from albert_zh.extract_feature import BertVector

# 加载训练好的模型
model = load_model("event_type.h5", custom_objects={"Attention": Attention})
with open("event_type.json", "r", encoding="utf-8") as f:
    event_type_list = json.loads(f.read())
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)


# 对单句话进行预测
def predict_single_text(text):
    # 将句子转换成向量
    vec = bert_model.encode([text])["encodes"][0]
    x_train = np.array([vec])

    # 模型预测
    predicted = model.predict(x_train)[0]
    indices = [i for i in range(len(predicted)) if predicted[i] > 0.5]
    one_hot = [0] * len(event_type_list)
    for index in indices:
        one_hot[index] = 1

    return one_hot, "|".join([event_type_list[index] for index in indices])


# 模型评估
def evaluate():
    with open("./data/multi-classification-test.txt", "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    true_y_list, pred_y_list = [], []
    true_label_list, pred_label_list = [], []
    common_cnt = 0
    for i in range(len(content)):
        print("predict %d samples" % (i+1))
        true_label, text = content[i].split(" ", maxsplit=1)
        true_y = [0] * len(event_type_list)
        for i, event_type in enumerate(event_type_list):
            if event_type in true_label:
                true_y[i] = 1

        pred_y, pred_label = predict_single_text(text)
        if set(true_label.split("|")) == set(pred_label.split("|")):
            common_cnt += 1
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)
        true_label_list.append(true_label)
        pred_label_list.append(pred_label)

    # F1值
    print(classification_report(true_y_list, pred_y_list, digits=4))
    return true_label_list, pred_label_list, hamming_loss(true_y_list, pred_y_list), common_cnt/len(true_y_list)


# 输出模型评估结果
true_labels, pred_lables, h_loss, accuracy = evaluate()
df = pd.DataFrame({"y_true": true_labels, "y_pred": pred_lables})
df.to_csv("pred_result.csv")

print("accuracy: ", accuracy)
print("hamming loss: ", h_loss)