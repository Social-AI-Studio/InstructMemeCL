# cp from lavis-main-4
1. 改成lora微调 
# done
2. 修改hidden state 的mask，把hidden state 和target 一样进行shift，然后再对应mask，再计算similarity
    新的是<yes/no>,<.>,<eos>，原来是<.>,<eos>,<>,原来的根本不能表达yes和no的meaning，对.和eos进行contrastive根本没用
# done

# cp from lavis-main-3
1. 去掉多余的东西 
# done
2. freeze掉bert，只train last 2 layers and output layer
# done
3. 实现step scale，每个epoch，loss scale不一样，在lavis/tasks/base_task.py里
# done

# cp from lavis-main-2
1. push to gitlab
# done
2. add online ranking for cluster loss
# done




# cp from lavis-main-1
1. push to gitlab 
# done
2. add contrastive loss: triplet loss
    1. add preprocessing_sampling
        1. random 
        2. preprocess
    # done
    2. add triplet loss
    # done
    3. add online ranking
    # todo






# cp from LAVIS-main
1. add cluster loss
    use_cluster bool
    output_hidden_states bool
    contrastive_layer
    cluster_margin
    cluster_metric
    sim_type: token, seq, maskedseq
# done


# todo
2. add triplet loss
    use_triplet bool
    contrastive_layer
    use_random_sampling

3. add negative/positive sampling online ranking method
    use_online_ranking

4. add preprocess negative/positive sampling method
    use_preprocess_sampling

output_hidden_states=True
if use_cluster:
    cluster_loss(contrastive_layer, cluster_margin, cluster_metric)
if use_triplet:
    if use_random_sampling:

    elif use_online_ranking:

    elif use_preprocess_sampling:

    else:
        raise error

