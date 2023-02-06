from typing import List, Tuple

import torch

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq import checkpoint_utils
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    GradMultiply,
)
from fairseq.utils import is_xla_tensor
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    TransformerEncoder,
    ConformerEncoder,
)
from fairseq.modules import LayerNorm
from fairseq.tasks import FairseqTask


@dataclass
class Wav2Vec2WordConfig(Wav2Vec2Config):
    ssl_checkpoint: str = field(
        default=MISSING,
        metadata={
            "help": "path to the cpc checkpoint"
        },
    )


@register_model("wav2vec2_word", dataclass=Wav2Vec2WordConfig)
class Wav2Vec2Word(Wav2Vec2Model):
    def __init__(self, cfg: Wav2Vec2WordConfig):
        BaseFairseqModel.__init__(self)

        ssl_model, ssl_cfg, ssl_task = checkpoint_utils.load_model_ensemble_and_task([cfg.ssl_checkpoint])

        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ssl_model[0].feature_extractor
        self.post_extract_proj = ssl_model[0].post_extract_proj
        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = ssl_model[0].dropout_input
        self.dropout_features = ssl_model[0].dropout_input

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = ssl_model[0].quantizer
        self.input_quantizer = ssl_model[0].input_quantizer

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        self.project_q = ssl_model[0].project_q

        cfg.quantize_input = cfg.quantize_input
        self.input_quantizer = ssl_model[0].input_quantizer
        if not cfg.quantize_input:
            self.project_inp = ssl_model[0].input_quantizer

        self.mask_emb = ssl_model[0].mask_emb
        self.encoder = ssl_model[0].encoder
        self.layer_norm = ssl_model[0].layer_norm
        self.target_glu = ssl_model[0].target_glu
        self.final_proj = ssl_model[0].final_proj


    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        boundaries=None
    ):
        source=torch.cat(source,dim=0).cuda()

        return super().forward(
            source,
            padding_mask,
            mask,
            features_only,
            layer,
            mask_indices,
            mask_channel_indices,
            padding_count,
        )