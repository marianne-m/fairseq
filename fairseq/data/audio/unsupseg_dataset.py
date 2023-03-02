# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import List, Tuple, Optional, Any, Union
import numpy as np
import torchaudio
import torch
from collections import defaultdict
from pathlib import Path
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)


def load_boundaries(manifest_path, max_sentence_length):
    labels = defaultdict(list)
    with open(manifest_path) as buf:
        for line in buf:
            try:
                sid, start, end, _ = line.rstrip().split()
            except:
                print(manifest_path, line)
                sys.exit()

            start, end = float(start), float(end)
            if end > max_sentence_length:
                continue
            labels[sid].append((start, end))
    return labels


def load_sentences(manifest_path, boundaries=None):
    paths = dict()
    with open(manifest_path) as buf:
        for index, line in enumerate(buf):
            path = line.rstrip()
            sid = Path(path).stem
            if sid in boundaries:
                paths[index] = (path, sid)

    logger.info((
        f"{len(paths)} sentences were loaded."
    ))
    return paths


class UnsupsegDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,  # file with path to word embeddings npz
        cfg: dict,
        sample_rate: float,
        label_rate: float,
        max_sentence_length: int = 20, # seconds
    ):
        self.max_sentence_length = max_sentence_length
        self.label_rate = label_rate
        self.sample_rate = sample_rate
        self.max_wav_length = self.max_sentence_length * self.sample_rate
        
        logger.info('Reading boundaries')
        dir_manifest_path = os.path.dirname(manifest_path)
        subset = os.path.basename(manifest_path)
        bound_manifest_path = os.path.join(dir_manifest_path, 'bound_'+subset)
        self.boundaries = load_boundaries(bound_manifest_path, self.max_sentence_length)

        logger.info('Reading sentences')
        self.paths = load_sentences(manifest_path, boundaries=self.boundaries)
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {"channels": 1, "length": 0, "rate": self.sample_rate}

    def __getitem__(self, index):
        path, sid = self.paths[index]
        boundaries = self.boundaries[sid]
        source, sample_rate = torchaudio.load(path)
        assert sample_rate == self.sample_rate, sample_rate

        # padding the audio
        source = source.flatten()
        source = source[:int(self.max_wav_length)]
        padded_source = torch.zeros(self.max_wav_length)
        padded_source[:len(source)] = source
        padded_source = padded_source.reshape(1, -1)
        return {"id": index, "source": padded_source, "boundaries": boundaries}

    def __len__(self):
        return len(self.paths)

    def crop_to_max_size(self, wav, target_size):
        return wav, 0

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}
        sources = [s["source"] for s in samples]
        boundaries = [s["boundaries"] for s in samples]
        ids = torch.LongTensor([s["id"] for s in samples])
        net_input = {
            "source": sources,
            "boundaries": boundaries,
        }
        batch = {
            "id": ids,
            "net_input": net_input
        }

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return 1

    def ordered_indices(self):
        return np.random.permutation(len(self))
