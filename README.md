# MoEC

Code for paper - MoEC: Mixture of Expert Clusters https://arxiv.org/abs/2207.09094

Please follow [fairseq document](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) to data pre-processing.

## Setup

Build:
```
pip install --user -e fairseq/
pip install --user -e infinibatch/
pip install -U numpy
```

## Data Pre-processing
We take machine translation as an example.

```bash
# Download and prepare the data
cd examples/translation/
# WMT'17 data:
bash prepare-wmt14en2de.sh
# or to use WMT'14 data:
# bash prepare-wmt14en2de.sh --icml17
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```



## Training
```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/wmt17_en_de_data/ \
        --save-dir /path/moec64/ckpt \
        --tensorboard-logdir /path/moec64/tb_logs \
        --log-format simple  --log-file /path/moec64/train.log \
        --arch gdmoe_wmt_en_de \
        --encoder-normalize-before \
        --task translation \
        --truncate-source \
        --max-source-positions 256 \
        --max-target-positions 256 \
        --criterion label_smoothed_cross_entropy_moe --label-smoothing 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
        --lr-scheduler inverse_sqrt --lr 5e-04 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 250 \
        --max-update 32000 \
        --attention-dropout 0.1 --dropout 0.3 \
        --max-tokens 4096 --update-freq 16 \
        --seed 1 \
        --skip-invalid-size-inputs-valid-test --fp16 --fp16-no-flatten-grads \
        --ddp-backend=no_c10d \
        --token-shuffle --moe-gate-loss-wt 0.01  --moe-gate-loss-combine-method sum \
        --no-epoch-checkpoints --clip-norm 0.1 \
        --encoder-moe-layers 3 --decoder-moe-layers 3 \
        --moe-top1-expert \
        --moe-sublayers 3 \
        --moe-expert-count 64 \
        --moe-gating-use-fp32 --tmoe-routing-dim-reduction \
        --tmoe-routing-dim 32 \
        --tmoe-routing-hard-cosine \
        --moe-activation-dropout 0.0 --moe-dropout 0.0 \
        --capacity-factor 2 \
        --sharded-save \
        --group-num 8 --exp-level-drop 0.5  --dropout-interval 250 --var-coef 1.0 --coef-type 1
        
```

## Inference
```
python -m torch.distributed.launch --nproc_per_node=1 generate_moe.py /path/wmt17_en_de_data/  \
    --path /path/moec64/ckpt/checkpoint_best.pt \
    --arch gdmoe_wmt_en_de \
    --task translation \
    --batch-size 128 --beam 5 \
    --model-overrides "{'coef_type':'1','encoder_moe_layers':'3', 'decoder_moe_layers':'3', 'moe_top1_expert':True, 'moe_sublayers':3, 'moe_expert_count':64,  'tmoe_routing_dim':32}"
```
