import os
import json
import scipy.stats as stats

def retrieve_results(input_folder):
    auc, acc = [], []
    # for seed in ['seed40', 'seed41', 'seed42', 'seed43', 'seed44']:
    for seed in ['seed47', 'seed49', 'seed46', 'seed41', 'seed43', 'seed52', 'seed 53']:
    for seed in ['seed42', 'seed46', 'seed41', 'seed47', 'seed49', 'seed50']:
        seed_folder = os.path.join(input_folder, seed)

        subfolders = os.listdir(seed_folder)
        subfolders = [x for x in subfolders if x != ".DS_Store"]
        print(subfolders)
        subfolder = os.path.join(seed_folder, subfolders[0])
        input_file = os.path.join(subfolder, 'evaluate.txt')

        # Initialize variables to keep track of the best row
        best_row = None
        max_agg_metrics = float('-inf')

        # Read the file line by line
        with open(input_file, 'r') as file:
            for line in file:
                # Parse each line as JSON
                row = json.loads(line)
                
                # Check if the current row has a better agg_metrics value
                if row['agg_metrics'] > max_agg_metrics:
                    max_agg_metrics = row['agg_metrics']
                    best_row = row

        # Print the row with the best agg_metrics
        # print(best_row)
        auc.append(best_row['auc'])
        acc.append(best_row['acc'])
    
    print(auc)
    print(acc)

    return auc, acc

# Example data
model_a = retrieve_results("lavis/outputfhm_lora_cluster/ep15_margin1.5")
model_b = retrieve_results("fhm_lora_ep15")

# model_a = retrieve_results('harm_lora/dm_r16_ep10')
# model_b = retrieve_results('harm_lora_cluster/ep10_margin3_scale0.1')

# model_a = retrieve_results('mami_lora_ep15')
# model_b = retrieve_results('mami_cluster/ep15_margin3_scale0.1')

# Paired t-test
t_stat_auroc, p_value_t_auroc = stats.ttest_rel(model_a[0], model_b[0])
t_stat_acc, p_value_t_acc = stats.ttest_rel(model_a[1], model_b[1])

# Wilcoxon signed-rank test
w_stat_auroc, p_value_w_auroc = stats.wilcoxon(model_a[0], model_b[0])
w_stat_acc, p_value_w_acc = stats.wilcoxon(model_a[1], model_b[1])

print("Paired t-test p-value:", p_value_t_auroc, p_value_t_acc)
print("Wilcoxon signed-rank test p-value:", p_value_w_auroc, p_value_w_acc)