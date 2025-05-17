#!/bin/bash

# filepath: /home/zhouy/code/SKB-SCM/auto_test.sh

# 创建结果保存目录
RESULTS_DIR="./results"
Mode='train'
Load_Checkpoint=0 
Train_Iters=50  
Mod_Method='bpsk'
Channel_Use=128

mkdir -p $RESULTS_DIR

RESULT_DIR_PATH="$RESULTS_DIR/${Mod_Method}"
mkdir -p "$RESULT_DIR_PATH"

# 定义参数范围
SNR_VALUES=(-18 -12 -6 0 6 12 18)  # 不同的 SNR 值
AID_VALUES=(0.001 0.005 0.01 0.05 0.1 0.2 0.4 0.6 0.8 1 1.5 2 2.5 3 3.5 4 4.5 5)  # 不同的 aid_alpha 值
MISMATCH_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0) # 不同的 mismatch_level 值

# 定义每个 SNR 对应的 tradeoff_lambda 值
TRADEOFF_LAMBDA_VALUES=(70 70 70 30 20 2 0.5)

# 遍历所有参数组合
for i in "${!SNR_VALUES[@]}"; do
    SNR=${SNR_VALUES[$i]}
    TRADEOFF_LAMBDA=${TRADEOFF_LAMBDA_VALUES[$i]}  # 根据 SNR 动态设置 tradeoff_lambda

    echo "=============================================="
    echo "Starting experiments for SNR=$SNR, tradeoff_lambda=$TRADEOFF_LAMBDA"
    echo "=============================================="

    for AID in "${AID_VALUES[@]}"; do
        echo "  Processing aid_alpha=$AID"

        for MISMATCH in "${MISMATCH_VALUES[@]}"; do
            echo "    Processing mismatch_level=$MISMATCH"

            # 定义实验结果文件名（与 train.py 中的文件名格式一致）
            RESULT_FILE="$RESULTS_DIR/${Mod_Method}/CIFAR_SNR${SNR}_lam${TRADEOFF_LAMBDA}_Trans${Channel_Use}_${Mod_Method}_mis${MISMATCH}_aid${AID}_SKB.csv"   

            # 检查结果文件是否已存在
            if [ -f "$RESULT_FILE" ]; then
                echo "      Skipping experiment: Result already exists"
            else
                echo "      Running experiment: SNR=$SNR, aid_alpha=$AID, mismatch_level=$MISMATCH"

                # 运行实验
                python3 main.py \
                    --mode $Mode \
                    --train_iters $Train_Iters \
                    --load_checkpoint $Load_Checkpoint \
                    --mod_method $Mod_Method \
                    --channel_use $Channel_Use \
                    --result_path $RESULTS_DIR \
                    --snr_train $SNR \
                    --snr_test $SNR \
                    --aid_alpha $AID \
                    --mismatch_level $MISMATCH \
                    --tradeoff_lambda $TRADEOFF_LAMBDA 

                # 检查实验是否成功
                if [ $? -eq 0 ]; then
                    echo "      Experiment completed successfully"
                else
                    echo "      Experiment failed"
                fi
            fi
        done
    done

    echo "=============================================="
    echo "Finished experiments for SNR=$SNR"
    echo "=============================================="
done

echo "All experiments completed!"