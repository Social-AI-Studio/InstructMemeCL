import os
import numpy as np
import re

def find_eval_files(folder_path, prefix):
    eval_files = []
    for root, dirs, files in os.walk(folder_path):
        # import pdb
        # pdb.set_trace()
        for dir_0 in dirs:
            # import pdb
            # pdb.set_trace()
            if dir_0.startswith(prefix):
                target_dir = os.path.join(root,dir_0)
                for file in os.listdir(target_dir):
                    # import pdb
                    # pdb.set_trace()
                    eval_file=os.path.join(target_dir,file,"evaluate.txt")
                    eval_files.append(eval_file)

    return sorted(eval_files)

folder_path = "lavis/output/fhm_lora/fhm_lora_ep15"
# folder_path = "lavis/output/fhm_lora_cluster/ep15_margin1.5"
# folder_path = "lavis/output/fhm_lora_triplet"

# folder_path = "lavis/output/harm_lora/dm_r16_ep10"
# folder_path = "lavis/output/harm_lora_cluster/ep10_margin3_scale0.1"
# folder_path = "lavis/output/harm_lora_triplet/"

# folder_path = "lavis/output/mami_lora"
# folder_path = "lavis/output/mami_cluster/ep15_margin3_scale0.1"

prefix = "seed"
epoch_num = 10
eval_files = find_eval_files(folder_path, prefix)
dictlist = []

class Result:
    def __init__(self, epoch, auc, acc, agg):
        self.epoch = epoch
        self.auc = auc
        self.acc = acc
        self.agg = agg

auc=[]
acc=[]
agg=[]
results = []
for eval_file in eval_files:
    print(eval_file)
    match = re.search(r"seed(\d+)", eval_file)

    if match:
        seed_number = match.group(1)
        print(f"The seed number is: {seed_number}")
    else:
        print("No seed number found.")
    with open(eval_file, 'r') as file:
        lines = file.readlines()
        # assert len(lines) == epoch_num
        if len(lines) < epoch_num:
            print("ERROR")
            continue
        pattern = re.compile(r'\{.*\}')
        auclist = []
        acclist = []
        agglist = []
        # 遍历每一行并查找字典
        for line in lines:
            match = re.search(pattern, line)
            if match:
                dictionary_str = match.group()
                dictionary = eval(dictionary_str)

                auclist.append(dictionary['auc'])
                acclist.append(dictionary['acc'])
                agglist.append(dictionary['agg_metrics'])
        max_value = max(agglist)
        max_index = agglist.index(max_value)
        # print("max epoch:  ", max_index, "  auc: ", auclist[max_index], "  acc: ", acclist[max_index])

        auc.append(auclist[max_index])
        acc.append(acclist[max_index])
        agg.append(agglist[max_index])
        results.append(Result(seed_number, auclist[max_index], acclist[max_index], agglist[max_index]))

sorted_results = sorted(results, key=lambda x: x.agg, reverse=True)
for record in sorted_results:
    print("max epoch:  ", record.epoch, "  auc: ", record.auc, "  acc: ", record.acc)

print(sorted(auc, reverse=True)[:5])
print(sorted(acc, reverse=True)[:5])

auc = sorted(auc, reverse=True)[:5]
acc = sorted(acc, reverse=True)[:5]

acc_mean = np.mean(acc)
acc_stdev = np.std(acc)

auc_mean = np.mean(auc)
auc_stdev = np.std(auc)

max_agg = max(agg)
max_seed = agg.index(max_agg)
print("num_seeds", len(acc))
print("max_seed", max_seed)

print("acc_mean: ", round(acc_mean*100, 2),"(",round(acc_stdev*100, 2),")")
print("auc_mean: ", round(auc_mean*100, 2),"(",round(auc_stdev*100, 2),")")
print("acc_stdev: ", acc_stdev)
print("auc_stdev: ", auc_stdev)
