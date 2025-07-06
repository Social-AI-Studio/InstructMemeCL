# import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import re

# dictlist = []
# auclist = []
# acclist = []
# agglist = []

# with open('lavis/output/BLIP2/MemeClassification_harm/2023092823582/2.txt', 'r') as file:
#     lines = file.readlines()
#     pattern = re.compile(r'\{.*\}')
#     # 遍历每一行并查找字典
#     for line in lines:
#         match = re.search(pattern, line)
#         if match:
#             dictionary_str = match.group()
#             dictionary = eval(dictionary_str)
#             dictlist.append(dictionary)
#             print(dictionary)
#             auclist.append(dictionary['auc'])
#             acclist.append(dictionary['acc'])
#             agglist.append(dictionary['agg_metrics'])

# max_acc = max(acclist)
# max_auc = max(auclist)
# max_agg = max(agglist)
# max_acc_index = acclist.index(max_acc)
# max_auc_index = auclist.index(max_auc)
# max_agg_index = agglist.index(max_agg)
# print("best acc", max_acc_index, max_acc)
# print("best auc", max_auc_index, max_auc)
# print("best agg", max_agg_index, max_agg)

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import statistics

dictlist = []
auclist = []
acclist = []
agglist = []

with open('mean.txt', 'r') as file:
    lines = file.readlines()
    pattern = re.compile(r'\{.*\}')
    # 遍历每一行并查找字典
    for line in lines:
        match = re.search(pattern, line)
        if match:
            dictionary_str = match.group()
            dictionary = eval(dictionary_str)
            dictlist.append(dictionary)
            print(dictionary)
            auclist.append(dictionary['auc'])
            acclist.append(dictionary['acc'])
            agglist.append(dictionary['agg_metrics'])

acc_mean = np.mean(acclist)
acc_stdev = np.std(acclist)

auc_mean = np.mean(auclist)
auc_stdev = np.std(auclist)
print(acclist)
print(auclist)

print("acc_mean: ", acc_mean*100,"(",acc_stdev*100,")")
print("auc_mean: ", auc_mean*100,"(",auc_stdev*100,")")
print("acc_stdev: ", acc_stdev)
print("auc_stdev: ", auc_stdev)

