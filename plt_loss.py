import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

dictlist = []
auclist = []
acclist = []
agglist = []

with open('paras/layer8.txt', 'r') as file:
    lines = file.readlines()
    # 遍历每一行并查找字典
    results = []
    for line in lines:
        result = re.search(r'loss: (\d+)', line)
        if result:
            # import pdb
            # pdb.set_trace()
            results.append(float(result.group(1)))
print("gap iter size: ", len(results))

plt.figure(1)
plt.title('loss layer8 result')

plt.plot(np.arange(len(results)), results, label="loss", color='k', linestyle='--')  

plt.legend()
plt.xlabel("gap iters")
plt.ylabel("loss")
plt.savefig('loss_layer8.png')
