#!/bin/bash

DATA=tokenized
CP=checkpoints/checkpoint_best.pt

python $(which fairseq-generate) $DATA --dataset-impl raw \
        --path $CP --gen-subset test \
        --beam 5 --batch-size 128 --lang-pairs ca-it,it-ca,ca-ro,ro-ca,ca-oc,oc-ca \
        --source-lang ca --target-lang ro \
        --task multilingual_translation_from_pretrained_xlm \
        --user-dir xlmr-finetuning/fairseq_model \
        --skip-invalid-size-inputs-valid-test --decoder-langtok --remove-bpe=sentencepiece
