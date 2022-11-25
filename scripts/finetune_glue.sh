#!/usr/bin/env bash

set -ex

# args: checkpoint_folder step

TASK=$1
BERT_MODEL_PATH=$2
GLUE_DATA_DIR=$3
OUTPUT_PATH=$4
ARCH=$5
N_EPOCH=$6
WARMUP_RATIO=$7
BSZ=$8
LR=$9
SEED=${10}
extra_args=${11}

BETAS="(0.9,0.98)"
CLIP=0.0
WEIGHT_DECAY=0.01

MAX_TOKENS=2200

echo $BERT_MODEL_PATH
# if [ ! -e $BERT_MODEL_PATH ]; then
#     echo "Checkpoint doesn't exist"
#     exit 0
# fi

TASK_DATA_DIR=$GLUE_DATA_DIR/$TASK-bin
OPTION=""

METRIC=accuracy
N_CLASSES=2
task_type=SMALL
if [ "$TASK" = "MNLI" ]
then
N_CLASSES=3
OPTION="--valid-subset valid,valid1"
EPOCH_ITER=12452
task_type=LARGE
fi

if [ "$TASK" = "QNLI" ]
then
EPOCH_ITER=3320
task_type=LARGE
fi

if [ "$TASK" = "QQP" ]
then
EPOCH_ITER=11392
task_type=LARGE
fi

if [ "$TASK" = "SST-2" ]
then
EPOCH_ITER=2105
task_type=LARGE
fi

if [ "$TASK" = "MRPC" ]
then
EPOCH_ITER=115
fi

if [ "$TASK" = "RTE" ]
then
EPOCH_ITER=101
fi

if [ "$TASK" = "CoLA" ]
then
METRIC=mcc
EPOCH_ITER=268
fi

if [ "$TASK" = "STS-B" ]
then
METRIC=pearson_spearman
N_CLASSES=1
OPTION="--regression-target"
EPOCH_ITER=180
fi

if [ "$task_type" = "LARGE" ]
then
    if [ "$N_EPOCH" = "10" ]
    then
        echo 'skip'
        exit 0
    fi
    # if [ "$WARMUP_RATIO" = "10" ]
    # then
    #     echo 'skip'
    #     exit 0
    # fi
    if [ "$BSZ" = "16" ]
    then
        echo 'skip'
        exit 0
    fi
fi

EPOCH_ITER=$((EPOCH_ITER*2)) # expand to itr for bsz=16
BSZ_EXPAND=$((BSZ/16))
MAX_TOKENS=$((MAX_TOKENS*BSZ/16)) # expand to itr for bsz=16

EPOCH_ITER=$((EPOCH_ITER/BSZ_EXPAND))
TOTAL_STEPS=$((EPOCH_ITER*N_EPOCH))
WARMUP_STEPS=$((TOTAL_STEPS/WARMUP_RATIO))
VALIDATE_INTERVAL=$((EPOCH_ITER/2))


echo $TASK_DATA_DIR
echo $EPOCH_ITER
echo $TOTAL_STEPS
echo $WARMUP_STEPS
echo $BSZ

OUTPUT_PATH=$OUTPUT_PATH/$TASK/$N_EPOCH-$WARMUP_RATIO-$BSZ-$LR-$SEED
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
if [ -e $OUTPUT_PATH/train_log.txt ]; then
    if grep -q 'done training' $OUTPUT_PATH/train_log.txt && grep -q 'Loaded checkpoint' $OUTPUT_PATH/train_log.txt; then
        echo "Training log existed"
        exit 0
    fi
fi

python train.py $TASK_DATA_DIR --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --restore-file $BERT_MODEL_PATH \
    --max-positions 512 \
    --max-sentences $BSZ \
    --max-tokens $MAX_TOKENS \
    --update-freq 1 \
    --task glue \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 8 \
    --init-token 0 --separator-token 2 \
    --arch $ARCH \
    --criterion sentence_prediction $OPTION \
    --num-classes $N_CLASSES \
    --dropout 0.1 --attention-dropout 0.1  --pooler-dropout 0.1 \
    --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "$BETAS" --adam-eps 1e-06 \
    --clip-norm $CLIP \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_STEPS --warmup-updates $WARMUP_STEPS  \
    --max-update $TOTAL_STEPS --seed $SEED --save-dir $OUTPUT_PATH --tensorboard-logdir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-save \
    --find-unused-parameters --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric $METRIC --maximize-best-checkpoint-metric --validate-interval-updates $VALIDATE_INTERVAL ${extra_args} |& tee $OUTPUT_PATH/train_log.txt
