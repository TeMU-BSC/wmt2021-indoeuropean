#!/bin/bash

    # Fairseq parameters
    DATA_DIR=data-bin
    TOTAL_UPDATES=125000    # Total number of training steps
    WARMUP_UPDATES=1000    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512  # Max sequence length
    MAX_POSITIONS=512      # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=4         # Number of sequences per batch (batch size)
    UPDATE_FREQ=8          # Increase the batch size 32x
    CP_DIR=checkpoints/xlmr.base # the folder with the reinitialized checkpoint
    CP="model.pt"   # сheckpoint for initializing
    
    $(which fairseq-train) --memory-efficient-fp16 $DATA_DIR \
    --task multilingual_translation_from_pretrained_xlm --criterion label_smoothed_cross_entropy --init-encoder-only --decoder-layers 3 \
    --lang-pairs ca-it,it-ca,ca-ro,ro-ca,ca-oc,oc-ca --arch multilingual_transformer_from_pretrained_xlm \
    --pretrained-xlm-checkpoint $CP_DIR/$CP --share-encoders --share-decoders --decoder-langtok --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --distributed-no-spawn --tensorboard-logdir tb --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1000 --user-dir xlmr-finetuning/fairseq_model \
    --save-dir checkpoints --ddp-backend no_c10d
