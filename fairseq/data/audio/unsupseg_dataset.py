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
from fairseq.data.fairseq_dataset import FairseqDataset
logger = logging.getLogger(__name__)


def load_boundaries(manifest_path, feat_sr, max_sentence_length):
    labels = {}
    with open(manifest_path) as buf:
        for line in buf:
            try:
                sid, start, end = line.rstrip().split(' ')[:3]
            except:
                print(manifest_path, line)
                sys.exit()
            start, end = float(start), float(end)
            if end > max_sentence_length:
                continue
            # start,end=np.around((feat_sr*start,feat_sr*end),0)
            if sid not in labels:
                labels[sid] = []
            labels[sid].append((start, end))
            # labels[sid].append((int(start),int(end)))
    return labels


def load_sentences(manifest_path, non_speech=False, boundaries=None):
    nb_sentences = 0
    paths = {}
    with open(manifest_path) as buf:
        for line in buf:
            path = line.rstrip()
            sid = path.split('/')[7].split('.')[0]
            if sid not in boundaries:
                continue
            paths[nb_sentences] = (path, sid)
            nb_sentences += 1
    logger.info((
        f"nb_sentences={nb_sentences} "
    ))
    return paths


class UnsupsegDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,  # file with path to word embeddings npz
        cfg: dict,
    ):
        self.max_sentence_length = 20  # seconds
        self.feat_sr = 50
        self.sr = 16000
        self.max_wav_length = self.max_sentence_length*self.sr
        logger.info('Reading boundaries')

        dir_manifest_path = os.path.dirname(manifest_path)
        subset = os.path.basename(manifest_path)
        bound_manifest_path = os.path.join(dir_manifest_path, 'bound_'+subset)
        self.boundaries = load_boundaries(bound_manifest_path, self.feat_sr, self.max_sentence_length)

        logger.info('Reading sentences')
        self.paths = load_sentences(manifest_path, boundaries=self.boundaries)
        self.slowest = 1.0
        self.src_info = {"rate": self.sr}
        self.target_info = {"channels": 1, "length": 0, "rate": self.sr}

    def __getitem__(self, index):
        path, sid = self.paths[index]
        boundaries = self.boundaries[sid]
        source, sr = torchaudio.load(path)
        assert sr == self.sr, sr
        source = source.flatten()
        # extract self.max_frames_sentences from source and labels
        source = source[:int(self.max_wav_length)]
        padded_source = torch.zeros(int(self.max_wav_length/self.slowest))
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
