# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Dict, Optional, Union
from collections import defaultdict
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.hubert_dataset import HubertDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)
import io

logger = logging.getLogger(__name__)


def load_audio_and_km_labels(
        manifest_path: str,
        label_path: str,
        max_keep: Optional[int],
        min_keep: Optional[int]
) -> dict:
    with open(label_path) as label_file:
        labels = [line.strip() for line in label_file]

    with open(manifest_path) as manifest_file:
        lines = [line.strip().split("\t") for line in manifest_file]
    
    root = lines[0]
    audio_files = lines[1:]

    assert(
        len(labels) == len(audio_files)
    ), f"number of labels does not match ({len(labels)} != {len(audio_files)})"

    files = dict()
    n_long, n_short = 0, 0
    for index, (items, label) in enumerate(zip(audio_files, labels)):
        assert len(items) == 2
        sz = int(items[1])
        if min_keep and sz < min_keep:
            n_short += 1
        elif max_keep and sz > max_keep:
            n_long += 1
        files[Path(items[0]).stem] = {
            "size": sz,
            "label": label,
            "filename": items[0]
        }

    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(audio_files)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max([val['size'] for val in files.values()])}, shortest-loaded={min([val['size'] for val in files.values()])}"
        )
    )
    return root, files


def load_boundaries(bound_path: List, data: Dict) -> defaultdict:
    boundaries = defaultdict(list)
    with open(bound_path) as f:
        for line in f:
            name, start, end, _ = line.rstrip().split()
            if name in data.keys():
                boundaries[name].append([float(start), float(end)])

    if len(boundaries) != len(data):
        logger.info(f"Number of word boundaries files does not match with "
                    f"number of audio files ({len(boundaries)} != {len(data)})")

    return boundaries


def verify_label_lengths(
    data: Dict,
    label_rate: int,
    audio_rate: int,
    label_path: str,
    tol: float = 0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    num_invalid = 0
    for file_id, info in data.items():
        dur_from_audio = info["size"] / audio_rate
        dur_from_label = len(info["label"].split()) / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"for file {file_id}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {info['size']}; "
                    f"label length = {len(info['label'])}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


def keep_files_with_all_data(data: Dict, boundaries: Dict) -> Dict:
    file_ids_boundaries = set(boundaries.keys())
    file_ids_data = set(data.keys())
    keepers = file_ids_boundaries.intersection(file_ids_data)
    return {ind: keep for ind, keep in enumerate(keepers)}


class HubertDatasetWB(HubertDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
    ) -> None:

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.num_labels = len(label_paths) #1
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )

        self.store_labels = True

        if self.num_labels != 1:
            raise NotImplementedError("Case where self.num_labels != 1 is not handled.")

        self.audio_root, self.data = load_audio_and_km_labels(
            manifest_path, label_paths[0], max_keep_sample_size, min_keep_sample_size
        )

        assert label_processors is None or len(label_processors) == self.num_labels
        verify_label_lengths(self.data, self.label_rates[0], self.sample_rate, label_paths[0])

        self.max_sample_size = (
            max_sample_size if max_sample_size else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

        subset = os.path.basename(manifest_path)
        self.bound_manifest_dir = os.path.join(os.path.dirname(manifest_path), 'bound_' + subset)
        self.boundaries = load_boundaries(self.bound_manifest_dir, self.data)

        index_names = keep_files_with_all_data(self.data, self.boundaries)

        self.audio_names = [self.data[name]["filename"] for name in index_names.values()]
        self.label_list = [[self.data[name]["label"] for name in index_names.values()]]
        self.boundaries_list = [self.boundaries[name] for name in index_names.values()]

    def get_boundaries(self, index):
        return self.boundaries_list[index]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        boundaries = self.get_boundaries(index)
        return {"id": index, "source": wav, "label_list": labels, "boundaries": boundaries}