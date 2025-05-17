#!/bin/bash

# filepath: /home/zhouy/code/SKB-SCM/auto_test_parallel.sh

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

# 最大并行任务数
MAX_JOBS=4

# 计算总任务数
TOTAL_TASKS=$(( ${#SNR_VALUES[@]} * ${#AID_VALUES[@]} * ${#MISMATCH_VALUES[@]} ))
COMPLETED_TASKS=0

# 定义一个函数，用于运行单个实验
run_experiment() {
    local SNR=$1
    local TRADEOFF_LAMBDA=$2
    local AID=$3
    local MISMATCH=$4

    echo "Running experiment with SNR=$SNR, TRADEOFF_LAMBDA=$TRADEOFF_LAMBDA, AID=$AID, MISMATCH=$MISMATCH"

    # 定义实验结果文件名
    RESULT_FILE="$RESULTS_DIR/${Mod_Method}/CIFAR_SNR${SNR}_lam${TRADEOFF_LAMBDA}_Trans${Channel_Use}_${Mod_Method}_mis${MISMATCH}_aid${AID}_SKB.csv"

    # 检查结果文件是否已存在
    if [ -f "$RESULT_FILE" ]; then
        echo "Skipping experiment: Result already exists"
    else
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
    fi

    # 增加已完成任务计数并显示进度
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
    echo "Progress: $COMPLETED_TASKS / $TOTAL_TASKS tasks completed ($(awk "BEGIN {printf \"%.2f\", ($COMPLETED_TASKS/$TOTAL_TASKS)*100}")%)"
}

# 遍历所有参数组合并运行任务
job_count=0
for i in "${!SNR_VALUES[@]}"; do
    SNR=${SNR_VALUES[$i]}
    TRADEOFF_LAMBDA=${TRADEOFF_LAMBDA_VALUES[$i]}  # 根据 SNR 动态设置 tradeoff_lambda

    echo "=============================================="
    echo "Starting experiments for SNR=$SNR, tradeoff_lambda=$TRADEOFF_LAMBDA"
    echo "=============================================="

    for AID in "${AID_VALUES[@]}"; do
        for MISMATCH in "${MISMATCH_VALUES[@]}"; do
            # 启动子进程运行任务
            run_experiment "$SNR" "$TRADEOFF_LAMBDA" "$AID" "$MISMATCH" &

            # 增加任务计数
            job_count=$((job_count + 1))

            # 如果达到最大并行任务数，则等待所有子进程完成
            if (( job_count >= MAX_JOBS )); then
                wait
                job_count=0
            fi
        done
    done
done

# 等待所有剩余的子进程完成
wait

echo "All experiments completed!"