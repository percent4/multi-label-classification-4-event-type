# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-04-09 21:31

from collections import defaultdict
from pprint import pprint

with open("./data/multi-classification-train.txt", "r", encoding="utf-8") as f:
    content = [_.strip() for _ in f.readlines()]

# 每个事件类型的数量统计
event_type_count_dict = defaultdict(int)

# 多事件类型数量
multi_event_type_cnt = 0

for line in content:
    # 事件类型
    event_types = line.split(" ", maxsplit=1)[0]

    # 如果|在事件类型中，则为多事件类型
    if "|" in event_types:
        multi_event_type_cnt += 1

    # 对应的每个事件类型数量加1
    for event_type in event_types.split("|"):
        event_type_count_dict[event_type] += 1


# 输出结果
print("多事件类型的样本共有%d个，占比为%.4f。" %(multi_event_type_cnt, multi_event_type_cnt/len(content)))

pprint(event_type_count_dict)