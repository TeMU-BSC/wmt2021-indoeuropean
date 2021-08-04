# Transfer Learning with Shallow Decoders: BSC at WMT2021’s Multilingual Low-Resource Translation for Indo-European Languages Shared Task

This repository contains the files that the BSC's team used for creating a submission for [Shared Task: Multilingual Low-Resource Translation for Indo-European Languages](http://www.statmt.org/wmt21/multilingualHeritage-translation-task.html). We only participated in Task 2, Romance Languages, with translations from Catalan to Occitan, Romanian and Italian.

The proposed method is described fully in the paper "Transfer Learning with Shallow Decoders: BSC at WMT2021’s Multilingual Low-Resource Translation for Indo-European Languages Shared Task" (link will be provided after publishing). All coding is implemented in [Fairseq==0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0/) and bash scripts are available for all the phases (tokenizing, preprocessing, finetuning, generating translations of valid and test).

The main idea is based on initializing a [multilingual transformer](https://github.com/pytorch/fairseq/tree/master/examples/translation#multilingual-translation) for all required directions with a shared encoder, shared decoder and shared embedding tables by [XLM-Roberta BASE pretrained language model](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) on the encoder side and a random shallow decoder of 3 layers. The code that we provide extend the available `fairseq/models/transformer_from_pretrained_xlm.py` with a multilingual functionality.

After concatenating all the datasets, we pre-process them and we finetune the above described multilingual transformer on available parallel data.

The pretrained model XLM-Roberta BASE, SentencePiece tokenizer and other necesary files can be downloaded with a command:

    # Download xlmr.base model
    wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz
    tar -xzvf xlmr.base.tar.gz

## Towards your MT model

1. Ensure there's no train_test_overlap

`clean_train_test_overlap.py`

2. Tokenize the train, valid and test sets

`tokenize.sh`

3. Preprocess the data

`preprocess-data-test.sh`

`preprocess-data.sh`

4. Finetune XLMR

`finetune-xlmr-encoder-only.sh` 

5. Generate translations

`generate-ca-it-test-xlmr-gpu.sh`

`generate-ca-it-valid-xlmr-gpu.sh`

`generate-ca-oc-test-xlmr-gpu.sh`

`generate-ca-oc-valid-xlmr-gpu.sh`

`generate-ca-ro-test-xlmr-gpu.sh`

`generate-ca-ro-valid-xlmr-gpu.sh`

6. Generate XML delivery files 

`make_xml_delivery.py`
