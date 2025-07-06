ulimit -n 12288

for warmup_steps in 0 500 1000
do
    python -u  main.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --train_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_train.json \
        --test_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_test.json \
        --img_dir /mnt/data1/datasets/memes/fhm_finegrained/images/img \
        --seed 40 \
        --warmup_steps $warmup_steps \
        --use_contrastive_loss \
        --margin 1.5 \
        --scale 1.0 \
        --metric cos \
        --train \
        --evaluate
done