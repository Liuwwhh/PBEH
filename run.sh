#!/bin/bash
set -e
DEVICE=0
# DEVICE=1
# DEVICE=2
# dataset_name=(MSCOCO_NoMean NUSWIDE_NoMean)
dataset_name=(MSCOCO MSCOCO_NoMean)
# dataset_name=(NUSWIDE NUSWIDE_NoMean)
# dataset_name=(MSCOCO_NoMean)
# dataset_name=(NUSWIDE_NoMean)
# dataset_name=(MSCOCO)
# dataset_name=(NUSWIDE)
# bit=(16 32 64 128 256)
# bit=(128 256)
bit=(16)
# prompt_mode=(share specific)
prompt_mode=(share)
# prompt_mode=(specific)

extend_bit=(1)

# error_samples_ratio=(0 0.02 0.04 0.06 0.08 0.1)
error_samples_ratio=(0.02) # 1.0表示不扩展哈希码

disstill_loss=(100)     # MSCOCO
learning_rate=(0.0001)      # MSCOCO
prompt_extend_per_sample_number=(800)   # MSCOCO

# disstill_loss=(200)     # NUSWIDE
# learning_rate=(0.00001)   # NUSWIDE
# prompt_extend_per_sample_number=(400)  # NUSWIDE

# MSCOCO disstill_loss 100 learning_rate 0.0001 prompt_extend_per_sample_number 800
# NUSWIDE disstill_loss 200 learning_rate 0.00001 prompt_extend_per_sample_number 400

# runid=(1)
# runid=1 完整版测试结果

# runid=(2)
# runid=(2) 取消哈希码扩展结果，即在error_samples_ratio=1.0的情况下

# runid=(3)
# # runid=(3) 在error_samples_ratio=0.1 & extend_bit=(1)的情况下，消融汉明空间约束，将汉明空间约束损失超参数设为0
# radius_constraint_loss=(0.01 0.1 1.0 10 100)
# # radius_constraint_loss=(0.0) # 0.0表示不使用汉明空间约束损失

# runid=(4)
# # runid=(4) 在error_samples_ratio=0 & extend_bit=(1)的情况下，
# prompt_hash_main_loss=(0.01 0.1 1.0 10 100)
# # prompt_hash_main_loss=(0.0) # 0.0

# runid=(5)
# # disstill_loss超参数敏感性实验
# disstill_loss=(100 200 300 400 500 600 700 800 900 1000)

# runid=(6)
# # prompt_extend_per_sample_number超参数实验
# prompt_extend_per_sample_number=(100 200 300 400 500 600 700 800 900 1000)

# runid=(7)
# runid=(7) 在error_samples_ratio=(0) & extend_bit=(1)的情况下，消融contrastive loss，将contrastive loss超参数设为0
# contras_loss=(0.0) # 0.0表示不使用contrastive loss

# runid=(8)
# # runid=(8) 在error_samples_ratio=(0) & extend_bit=(1)的情况下，消融disstill_loss，将disstill_loss超参数设为0
# disstill_loss=(0.0) # 0.0表示不使用contrastive loss

# runid=(9)
# # runid=(9) 在error_samples_ratio=(0) & extend_bit=(1)的情况下，测试contras_loss参数敏感性
# contras_loss=(0.0 0.01 0.1 1.0 10 100)

# runid=(10)
# # # runid=(10) 在error_samples_ratio=(0) & extend_bit=(1)的情况下，测试hash_cos_sim_loss参数敏感性
# # hash_cos_sim_loss=(0.0 0.01 0.1 1.0 10 100)
# hash_cos_sim_loss=(0.0)

# runid=(11)
# # 测试similartity_main_loss参数敏感性
# similartity_main_loss=(0.01 0.1 1.0 10 100 200 300) # 0.01表示不使用similarity main loss
# # similartity_main_loss=(0) # 0.01表示不使用similarity main loss

# runid=(12)
# # 测试binary_similartity_main_loss参数敏感性
# # binary_similartity_main_loss=(0.01 0.1 1.0 10 100)
# binary_similartity_main_loss=(0.0) # 0.0表示不使用binary similarity main loss

# runid=(13)
# # 测试quantify_loss参数敏感性
# quantify_loss=(0.01 0.1 1.0 10 100)
# # quantify_loss=(0.0) # 0.0表示不使用quantify loss

# runid=(14)
# # 测试learning_rate参数敏感性
# learning_rate=(0.000001 0.00001 0.0001 0.001 0.01 0.1)

# runid=(15)
# 测试相同学习率情况下的影响（消融实验，消掉不对称优化）
# 修改extend_learning_rate

runid=(17)
# runid=(3) 在error_samples_ratio=0.02 & extend_bit=(1)的情况下，消融汉明空间约束，将汉明空间约束损失超参数设为0
radius_constraint_loss=(0.01 0.1 1.0 10 100)
# radius_constraint_loss=(0.0) # 0.0表示不使用汉明空间约束损失

for s in ${dataset_name[@]}
    do
    for n in ${prompt_mode[@]}
        do
        for d in ${prompt_extend_per_sample_number[@]}
            do
            for b in ${bit[@]}
                do
                for e in ${extend_bit[@]}
                    do
                    for l in ${learning_rate[@]}
                        do
                        for r in ${disstill_loss[@]}
                            do
                            for a in ${error_samples_ratio[@]}
                                do
                                for run in ${runid[@]}
                                    do
                                    for radius in ${radius_constraint_loss[@]}
                                        do
                                        echo "Running with dataset: $s, prompt_mode: $n, bit: $b, extend_per_sample_number: $d, extend_bit: $e, learning_rate: $l, disstill_loss: $r, error_samples_ratio: $a, runid: $run, radius_constraint_loss: $radius"
                                        CUDA_VISIBLE_DEVICES=$DEVICE python /data/liuwenhao/AE_PrePrain_OnlyClassifier_AddCenter_ModifyLoss/main.py --bit $b --dataset_name $s --prompt_mode $n --prompt_extend_per_sample_number $d --extend_bit $e --learning_rate $l --disstill_loss $r --error_samples_ratio $a --runid $run --radius_constraint_loss $radius
                                        done
                                    # echo "Running with dataset: $s, prompt_mode: $n, bit: $b, extend_per_sample_number: $d, extend_bit: $e, learning_rate: $l, disstill_loss: $r, error_samples_ratio: $a, runid: $run"
                                    # CUDA_VISIBLE_DEVICES=$DEVICE python /data/liuwenhao/AE_PrePrain_OnlyClassifier_AddCenter_ModifyLoss/main.py --bit $b --dataset_name $s --prompt_mode $n --prompt_extend_per_sample_number $d --extend_bit $e --learning_rate $l --disstill_loss $r --error_samples_ratio $a --runid $run
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
