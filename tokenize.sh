#!/bin/sh

python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/test/test.ca --outputs tokenized/test.ca
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/valid/valid.ca --outputs tokenized/valid.ca
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/valid/valid.it --outputs tokenized/valid.it
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/valid/valid.oc --outputs tokenized/valid.oc
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/valid/valid.ro --outputs tokenized/valid.ro
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-it/train.ca-it.ca --outputs tokenized/train.ca-it.ca --max-len 512
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-it/train.ca-it.it --outputs tokenized/train.ca-it.it --max-len 512
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-oc/train.ca-oc.ca --outputs tokenized/train.ca-oc.ca --max-len 512
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-oc/train.ca-oc.oc --outputs tokenized/train.ca-oc.oc --max-len 512
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-ro/train.ca-ro.ca --outputs tokenized/train.ca-ro.ca --max-len 512
python3 spm_encode.py --model sentencepiece.bpe.model --inputs raw/train/ca-ro/train.ca-ro.ro --outputs tokenized/train.ca-ro.ro --max-len 512
