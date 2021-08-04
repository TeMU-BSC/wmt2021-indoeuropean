#!/usr/bin/env bash

USER_DIR=xlmr-finetuning/fairseq_model
DATA_DIR=tokenized 
DEST_DIR=data-bin

mkdir $DEST_DIR

# preprocess CA-IT

SRC="ca"
TGT="it"

python $(which fairseq-preprocess) --source-lang $SRC \
  --target-lang $TGT \
  --trainpref "${DATA_DIR}/train.ca-it" \
  --validpref "${DATA_DIR}/valid" \
  --destdir $DEST_DIR \
  --workers 128 \
  --srcdict "${DATA_DIR}/dict.txt" \
  --tgtdict "${DATA_DIR}/dict.txt" \
  --task multilingual_translation_from_pretrained_xlm --user-dir $USER_DIR



python $(which fairseq-preprocess) --source-lang $TGT \
  --target-lang $SRC \
  --trainpref "${DATA_DIR}/train.ca-it" \
  --validpref "${DATA_DIR}/valid" \
  --destdir $DEST_DIR \
  --workers 128 \
  --srcdict "${DATA_DIR}/dict.txt" \
  --tgtdict "${DATA_DIR}/dict.txt" \
  --task multilingual_translation_from_pretrained_xlm --user-dir $USER_DIR

# preprocess CA-RO

SRC="ca"
TGT="ro"

python $(which fairseq-preprocess) --source-lang $SRC \
  --target-lang $TGT \
  --trainpref "${DATA_DIR}/train.ca-ro" \
  --validpref "${DATA_DIR}/valid" \
  --destdir $DEST_DIR \
  --workers 128 \
  --srcdict "${DATA_DIR}/dict.txt" \
  --tgtdict "${DATA_DIR}/dict.txt" \
  --task multilingual_translation_from_pretrained_xlm --user-dir $USER_DIR



python $(which fairseq-preprocess) --source-lang $TGT \
  --target-lang $SRC \
  --trainpref "${DATA_DIR}/train.ca-ro" \
  --validpref "${DATA_DIR}/valid" \
  --destdir $DEST_DIR \
  --workers 128 \
  --srcdict "${DATA_DIR}/dict.txt" \
  --tgtdict "${DATA_DIR}/dict.txt" \
  --task multilingual_translation_from_pretrained_xlm --user-dir $USER_DIR
  
# preprocess CA-OC

SRC="ca"
TGT="oc"

python $(which fairseq-preprocess) --source-lang $SRC \
  --target-lang $TGT \
  --trainpref "${DATA_DIR}/train.ca-oc" \
  --validpref "${DATA_DIR}/valid" \
  --destdir $DEST_DIR \
  --workers 128 \
  --srcdict "${DATA_DIR}/dict.txt" \
  --tgtdict "${DATA_DIR}/dict.txt" \
  --task multilingual_translation_from_pretrained_xlm --user-dir $USER_DIR



python $(which fairseq-preprocess) --source-lang $TGT \
  --target-lang $SRC \
  --trainpref "${DATA_DIR}/train.ca-oc" \
  --validpref "${DATA_DIR}/valid" \
  --destdir $DEST_DIR \
  --workers 128 \
  --srcdict "${DATA_DIR}/dict.txt" \
  --tgtdict "${DATA_DIR}/dict.txt" \
  --task multilingual_translation_from_pretrained_xlm --user-dir $USER_DIR
