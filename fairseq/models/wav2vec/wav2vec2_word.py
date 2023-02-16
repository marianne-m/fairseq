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
from random import sample
import numpy as np


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
        if isinstance(source, list):
            source=torch.cat(source,dim=0).cuda()

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            if boundaries is not None:
                mask_indices = self.compute_mask_indices_with_word_boundaries(features.shape, boundaries)
                mask_indices = torch.from_numpy(mask_indices).to(features.device)
            else:
                mask_indices = None

            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)
                y = y.detach()

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)
                y = y.detach()

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def compute_mask_indices_with_word_boundaries(
        self,
        shape: tuple,
        boundaries: List,
        mask_prob = 0.15
    ):
        bsz, size, _ = shape
        mask = np.full((bsz, size), False)
        mask_idcs = [[]] * bsz
        for batch_idx, b_boundaries in enumerate(boundaries):
            nb_of_masked_words = int(mask_prob*len(b_boundaries))
            masks = sample(b_boundaries, nb_of_masked_words)
            mask_indices = [[int(time/0.020) for time in mask] for mask in masks]
            for start, stop in mask_indices:
                mask_idcs[batch_idx].extend([i for i in range(start, stop)])
                mask[batch_idx, start:stop] = True

        # min_len = min([len(m) for m in mask_idcs])
        # for i, mask_idc in enumerate(mask_idcs):
        #     if len(mask_idc) > min_len:
        #         mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        #     mask[i, mask_idc] = True
   
        return mask