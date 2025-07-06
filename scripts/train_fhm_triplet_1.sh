# conda activate lavis-cit

# export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-automl/local/cuda-11.6/lib64/:$LD_LIBRARY_PATH
# export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-automl/local/cuda-11.6/bin:$PATH

# conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/gaozihan04/env/lavis_gzh_lora

# 

# Use CUDA 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

nvcc --version

seed=40

for seed in 41 42 43 44 45 46 47 48 49
do
    CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.run \
        --master_port=25671 \
        --nproc_per_node=1 train.py \
        --cfg-path fhm_lora_cluster_tripletloss/c_b16_ep10_lr10_r16_dm_seed${seed}.yaml
done