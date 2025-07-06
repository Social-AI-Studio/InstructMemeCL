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

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path debug.yaml