# Transfer Learning with Shallow Decoders: BSC at WMT2021’s Multilingual Low-Resource Translation for Indo-European Languages Shared Task

This repository contains the files that the BSC's team used for creating a submission for [Shared Task: Multilingual Low-Resource Translation for Indo-European Languages](http://www.statmt.org/wmt21/multilingualHeritage-translation-task.html). We only participated in the Task 2, Romance Languages, with translations from Catalan to Occitan, Romanian and Italian.

The proposed method is described fully in the paper "Transfer Learning with Shallow Decoders: BSC at WMT2021’s Multilingual Low-Resource Translation for Indo-European Languages Shared Task" (link will be provided after publishing). All coding is implemented in [Fairseq==0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0/) and bash scripts are available for all the phases (tokenizing, preprocessing, finetuning, generating translations of valid and test).

The main idea is based on initializing a [multilingual transformer](https://github.com/pytorch/fairseq/tree/master/examples/translation#multilingual-translation) for all required directions with a shared encoder, shared decoder and shared embedding tables by [XLM-Roberta BASE pretrained language model](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) on the encoder side and a random shallow decoder of 3 layers. The code that we provide extend the available `fairseq/models/transformer_from_pretrained_xlm.py` with a multilingual functionality.

After downloading the datasets and adding a small dataset ca-oc that we created from a crawling of the Catalan Government domains and subdomains we finetune the above described multilingual transfrmer on available parallel data.

The pretrained model XLM-Roberta BASE, SentencePiece tokenizer and other necesary files can be downloaded with a command:

    # Download xlmr.base model
    wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz
    tar -xzvf xlmr.base.tar.gz

