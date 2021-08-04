# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict
from typing import Any, Dict

from fairseq import checkpoint_utils, utils
from fairseq.checkpoint_utils import prune_state_dict
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture, FairseqMultiModel
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    Embedding,
    )

from fairseq.models.transformer import base_architecture as transformer_base_architecture

from fairseq.models.multilingual_transformer import MultilingualTransformerModel, base_multilingual_architecture

class CustomFairseqMultiModel(FairseqMultiModel):

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.
        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, args)
        return super().load_state_dict(new_state_dict, strict)

@register_model('custom_multilingual_transformer')
class CustomMultilingualTransformerModel(CustomFairseqMultiModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        custom_base_multilingual_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=args.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=args.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                    )
                lang_encoders[lang] = TransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                lang_decoders[lang] = TransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)

        return CustomMultilingualTransformerModel(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, args=None):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, args=args)


@register_model("multilingual_transformer_from_pretrained_xlm")
class MultilingualTransformerFromPretrainedXLMModel(CustomMultilingualTransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MultilingualTransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-xlm-checkpoint",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into encoder",
        )

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "You must specify a path for --pretrained-xlm-checkpoint to use "
            "--arch transformer_from_pretrained_xlm"
        )
        assert isinstance(task.source_dictionary, cls_dictionary) and isinstance(
            task.target_dictionary, cls_dictionary
        ), (
            "You should use a MaskedLMDictionary when using --arch "
            "transformer_from_pretrained_xlm because the pretrained XLM model "
            "was trained using data binarized with MaskedLMDictionary. "
            "For translation, you may want to use --task "
            "translation_from_pretrained_xlm"
        )
        assert not (
            getattr(args, "init_encoder_only", False)
            and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedXLM(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderFromPretrainedXLM(args, tgt_dict, embed_tokens)


def upgrade_state_dict_with_xlm_weights(
    state_dict: Dict[str, Any], pretrained_xlm_checkpoint: str
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_xlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_xlm_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_xlm_checkpoint)
    xlm_state_dict = state["model"]
    for key in xlm_state_dict.keys():

        for search_key in ["embed_tokens", "embed_positions", "layers"]:
            if search_key in key:
                subkey = key[key.find(search_key):]
                assert subkey in state_dict, (
                    "{} Transformer encoder / decoder "
                    "state_dict does not contain {}. Cannot "
                    "load {} from pretrained XLM checkpoint "
                    "{} into Transformer.".format(
                        str(state_dict.keys()),
                        subkey, key, pretrained_xlm_checkpoint)
                    )

                state_dict[subkey] = xlm_state_dict[key]
    return state_dict


class TransformerEncoderFromPretrainedXLM(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, 'init_decoder_only', False):
            # Don't load XLM weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "encoder from pretrained XLM"
        )
        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


class TransformerDecoderFromPretrainedXLM(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, 'init_encoder_only', False):
            # Don't load XLM weights for decoder if --init-encoder-only
            return
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "decoder from pretrained XLM"
        )

        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)

@register_model_architecture("custom_multilingual_transformer","custom_multilingual_transformer")
def custom_base_multilingual_architecture(args):
    transformer_base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)


@register_model_architecture(
    "multilingual_transformer_from_pretrained_xlm", "multilingual_transformer_from_pretrained_xlm"
)
def pretrained_xlm_base_architecture(args):
    custom_base_multilingual_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('custom_multilingual_transformer', 'multilingual_transformer_vaswani_wmt_en_de_big')
def multilingual_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    custom_base_multilingual_architecture(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture('custom_multilingual_transformer', 'multilingual_transformer_wmt_en_de_big_t2t')
def multilingual_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    multilingual_transformer_vaswani_wmt_en_de_big(args)