#!/bin/bash

python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-it/train.ca-it.ca --outputs tokenized/train.ca-it.ca --max-len 512
