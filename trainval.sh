#!/bin/bash

PHASE=$1  # train, eval
DATASET=$2  # jhmdb, ucf24

CFG_FILE="config_files/${DATASET}/openmixer_e2e.yaml"
# CFG_FILE="config_files/${DATASET}/openmixer_zsr_tl.yaml"
# CFG_FILE="config_files/${DATASET}/openmixer_zsr_zsl.yaml"  # eval-only!

TEST_WEIGHT=${3:-'checkpoints/model_final.pth'}

eval "$(conda shell.bash hook)"
conda activate stmixer_new

if [ $PHASE == 'train' ]
then
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=2024 train_net.py \
        --config-file ${CFG_FILE} \
        --transfer \
        --no-head \
        --use-tfboard
elif [ $PHASE == 'eval' ]
then
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=2405 test_net.py \
        --config-file ${CFG_FILE} \
        MODEL.WEIGHT ${TEST_WEIGHT}
fi

echo "${PHASE} finished!"