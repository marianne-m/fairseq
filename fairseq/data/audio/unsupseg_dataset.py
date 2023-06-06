# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import torchaudio
import torch
from collections import defaultdict
from pathlib import Path
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data import data_utils
from random import choice

logger = logging.getLogger(__name__)


def load_boundaries(
        manifest_path: str,
        max_sentence_length: int = None
) -> Dict[str, Tuple[int, int]]:
    labels = defaultdict(list)
    with open(manifest_path) as buf:
        for line in buf:
            try:
                sid, start, end, _ = line.rstrip().split()
            except:
                print(manifest_path, line)
                sys.exit()

            start, end = float(start), float(end)
            if max_sentence_length and end > max_sentence_length:
                continue
            labels[sid].append((start, end))
    return labels


def load_boundaries_and_transcript(
        manifest_path: str,
        max_sentence_length: int = None
) -> Dict[str, Tuple[int, int]]:
    labels = defaultdict(list)
    with open(manifest_path) as buf:
        for line in buf:
            try:
                sid, start, end, transcript = line.rstrip().split()
            except:
                print(manifest_path, line)
                sys.exit()

            start, end = float(start), float(end)
            if max_sentence_length and end > max_sentence_length:
                continue
            labels[sid].append(((start, end), transcript))
    return labels


def load_sentences(
        manifest_path: str,
        boundaries: Dict
) -> Dict[int, Tuple[str, str, int]]:
    """
    Loads the path, file id, and size for files already in boundaries
    """
    paths = dict()
    root = ""
    index = 0
    with open(manifest_path) as buf:
        for line in buf:
            if index == 0 and len(line.rstrip().split("\t")) == 1:
                root = line.strip()
                continue
            path, size = line.rstrip().split("\t")
            sid = Path(path).stem
            if sid in boundaries:
                paths[index] = (root + "/" + path, sid, size)
                index += 1

    logger.info((
        f"{len(paths)} sentences were loaded."
    ))
    return paths


def load_sentences_and_km_labels(
        manifest_path: str,
        label_path: str,
        boundaries: Dict
) -> Dict[int, Tuple[str, str, int, str]]:
    """
    Loads the path, file id, size, and labels for files already in boundaries
    """
    paths = dict()
    index = 0
    with open(manifest_path) as dir_path:
        with open(label_path) as file_labels:
            for line, label in zip(dir_path, file_labels):
                path, size = line.rstrip().split("\t")
                sid = Path(path).stem
                if sid in boundaries:
                    paths[index] = (path, sid, size, label)
                    index += 1

    logger.info((
        f"{len(paths)} sentences were loaded."
    ))
    return paths


class UnsupsegDatasetKmeans(FairseqDataset):
    """
    UnsupsegDataset to read Hubert kmeans labels and boundaries labels
    """
    def __init__(
        self,
        manifest_path: str,  # file with path to word embeddings npz
        sample_rate: float,
        label_path: str,
        label_rate: float,
        max_sentence_length: int = 20,  # seconds
    ):
        self.max_sentence_length = max_sentence_length
        self.label_rate = label_rate
        self.sample_rate = sample_rate
        self.max_wav_length = self.max_sentence_length * self.sample_rate
        self.max_frames_sentence = int(self.label_rate * self.max_sentence_length)

        logger.info('Reading boundaries')
        dir_manifest_path = os.path.dirname(manifest_path)
        subset = os.path.basename(manifest_path)
        bound_manifest_path = os.path.join(dir_manifest_path, 'bound_'+subset)
        self.boundaries = load_boundaries(bound_manifest_path, self.max_sentence_length)

        logger.info('Reading sentences')
        self.paths = load_sentences_and_km_labels(manifest_path, label_path, self.boundaries)
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {"channels": 1, "length": 0, "rate": self.sample_rate}

    def __getitem__(self, index):
        path, sid, _, labels = self.paths[index]

        labels = torch.tensor([int(label) for label in labels.split()])
        boundaries = self.boundaries[sid]
        source, sample_rate = torchaudio.load(path)
        assert sample_rate == self.sample_rate, sample_rate

        # padding the audio
        source = source.flatten()
        source = source[:int(self.max_wav_length)]
        padded_source = torch.zeros(self.max_wav_length)
        padded_source[:len(source)] = source
        padded_source = padded_source.reshape(1, -1)

        # padding mask
        padding_mask = torch.BoolTensor(self.max_wav_length)
        padding_mask[:len(source)] = False

        # padding the labels
        labels = labels[:self.max_frames_sentence]
        padded_labels = torch.zeros(self.max_frames_sentence)
        padded_labels[:len(labels)] = labels
        padded_labels = padded_labels.reshape(1, -1)

        return {
            "id": index,
            "source": padded_source,
            "padding_mask": padding_mask,
            "boundaries": boundaries,
            "label_list": padded_labels
        }

    def __len__(self):
        return len(self.paths)

    def crop_to_max_size(self, wav, target_size):
        return wav, 0

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = ([s["source"] for s in samples]) # = audios
        sources = torch.stack(sources).squeeze()
        padding_mask = ([s["padding_mask"] for s in samples])
        padding_mask = torch.stack(padding_mask).squeeze()
        boundaries = [s["boundaries"] for s in samples]
        target_list = [s["label_list"] for s in samples]
        target_list = [torch.stack(target_list).squeeze()]
        ids = torch.LongTensor([s["id"] for s in samples])

        net_input = {
            "source": sources,
            "boundaries": boundaries,
            "padding_mask": padding_mask
        }
        batch = {
            "id": ids,
            "net_input": net_input,
            "target_list": target_list
        }

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return 1

    def ordered_indices(self):
        return np.random.permutation(len(self))

class UnsupsegDataset(FairseqDataset):
    """
    This dataset can be used to finetune Wav2Vec2 for ASR and loads word boundaries labels
    """
    def __init__(
        self,
        manifest_path: str,  # file with path to word embeddings npz
        sample_rate: float,
        pad,
        max_sentence_length: int = 20,  # seconds
        process_label = None,
    ):
        self.max_sentence_length = max_sentence_length
        self.sample_rate = sample_rate
        self.max_wav_length = self.max_sentence_length * self.sample_rate

        logger.info('Reading boundaries')
        dir_manifest_path = os.path.dirname(manifest_path)
        subset = os.path.basename(manifest_path)
        bound_manifest_path = os.path.join(dir_manifest_path, 'bound_'+subset)
        self.boundaries = load_boundaries_and_transcript(bound_manifest_path)

        logger.info('Reading sentences')
        self.paths = load_sentences(manifest_path, self.boundaries)
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {"channels": 1, "length": 0, "rate": self.sample_rate}
        self.len_data = len(self.paths)

        self.process_label = process_label
        self.pad = pad

    def __getitem__(self, index):
        path, sid, _ = self.paths[index]

        boundaries = self.boundaries[sid]
        source, sample_rate = torchaudio.load(path)
        assert sample_rate == self.sample_rate, sample_rate

        # padding the audio
        source = source.flatten()
        source = source[:int(self.max_wav_length)]
        padded_source = torch.zeros(self.max_wav_length)
        padded_source[:len(source)] = source
        padded_source = padded_source.reshape(1, -1)

        # padding mask
        padding_mask = torch.BoolTensor(self.max_wav_length)
        padding_mask[:len(source)] = False

        # finding the transcript
        transcript = np.array([bound[1] for bound in boundaries])
        boundaries = np.array([bound[0] for bound in boundaries])
        transcript = transcript[(boundaries[:,0]) < self.max_sentence_length]
        transcript = [" ".join(word).upper() for word in transcript if word != 'SIL']
        boundaries = boundaries[boundaries[:,0] < self.max_sentence_length]
        transcript = " | ".join(transcript) + " |"

        if self.process_label is not None:
            transcript = self.process_label(transcript)

        return {
            "id": index,
            "source": padded_source,
            "padding_mask": padding_mask,
            "boundaries": boundaries.tolist(),
            # "boundaries_vector": word_boundaries,
            "target": transcript
        }

    def __len__(self):
        # nb de fois où getitem est appelée dans une epoch
        return len(self.paths)

    def crop_to_max_size(self, wav, target_size):
        return wav, 0

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = ([s["source"] for s in samples]) # = audios
        sources = torch.stack(sources).squeeze()
        padding_mask = ([s["padding_mask"] for s in samples])
        padding_mask = torch.stack(padding_mask).squeeze()
        boundaries = [s["boundaries"] for s in samples]
        ids = torch.LongTensor([s["id"] for s in samples])
        target = [s["target"] for s in samples]

        target_lengths = torch.LongTensor([len(t) for t in target])
        target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        ntokens = target_lengths.sum().item()

        net_input = {
            "source": sources,
            "boundaries": boundaries,
            "padding_mask": padding_mask
        }
        batch = {
            "id": ids,
            "net_input": net_input,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens
        }

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sample_rate * self.max_sentence_length


class UnsupsegMandarinDataset(FairseqDataset):
    """
    This dataset can be used to finetune Wav2Vec2 for ASR and loads word boundaries labels
    """
    def __init__(
        self,
        manifest_path: str,  # file with path to word embeddings npz
        sample_rate: float,
        pad,
        max_sentence_length: int = 20,  # seconds
        process_label = None,
    ):
        self.max_sentence_length = max_sentence_length
        self.sample_rate = sample_rate
        self.max_wav_length = self.max_sentence_length * self.sample_rate

        logger.info('Reading boundaries')
        dir_manifest_path = os.path.dirname(manifest_path)
        subset = os.path.basename(manifest_path)
        bound_manifest_path = os.path.join(dir_manifest_path, 'bound_'+subset)
        self.boundaries = load_boundaries_and_transcript(bound_manifest_path)

        logger.info('Reading sentences')
        self.paths = load_sentences(manifest_path, self.boundaries)
        self.src_info = {"rate": self.sample_rate}
        self.target_info = {"channels": 1, "length": 0, "rate": self.sample_rate}
        self.len_data = len(self.paths)

        self.process_label = process_label
        self.pad = pad

    def __getitem__(self, index):
        index = np.random.randint(self.len_data)
        path, sid, _ = self.paths[index]

        boundaries = self.boundaries[sid]
        source, sample_rate = torchaudio.load(path)
        assert sample_rate == self.sample_rate, sample_rate

        source = source.flatten()
        source_len = source.shape[0]

        # cropping the audio
        crop_onset = source_len
        max_onset = source_len - self.max_wav_length
        retry = 0
        while crop_onset > max_onset:
            onset_index = np.random.randint(len(boundaries))
            crop_onset_sec = boundaries[onset_index][0][0]
            crop_onset = int(crop_onset_sec * sample_rate)
            retry += 1
            if retry > 10:
                break                

        crop_offset = min(crop_onset + self.max_wav_length, source_len)
        crop_offset_sec = crop_onset_sec + self.max_sentence_length
        source = source[crop_onset:crop_offset]

        # retrieving the transcript & creating a vector with 1 where word boundaries are
        offset_index = onset_index
        last_bound = boundaries[offset_index][0][1]

        word_boundaries = np.zeros(self.max_wav_length, dtype=int)
        while offset_index < len(boundaries) and last_bound < crop_offset_sec:
            last_bound_start, last_bound = boundaries[offset_index][0]

            bound_index_start = int(last_bound_start * self.sample_rate) - crop_onset
            bound_index_end = int(last_bound * self.sample_rate) - crop_onset
            word_boundaries[max(bound_index_start - 1, 0): bound_index_start + 2] = 1
            word_boundaries[max(bound_index_end - 1, 0): bound_index_end + 2] = 1

            offset_index += 1

        transcript = [bound[1].replace(",", " ") for bound in boundaries[onset_index:offset_index]]
        transcript = " | ".join(transcript)
        transcript += " |"

        boundaries = [bound[0] for bound in boundaries[onset_index:offset_index]]

        if self.process_label is not None:
            transcript = self.process_label(transcript)

        # padding the audio
        padded_source = torch.zeros(self.max_wav_length)
        padded_source[:len(source)] = source
        padded_source = source
        padded_source = padded_source.reshape(1, -1)

        # padding mask
        padding_mask = torch.BoolTensor(self.max_wav_length)
        padding_mask[:len(source)] = False

        return {
            "id": index,
            "source": padded_source,
            "padding_mask": padding_mask,
            "boundaries": boundaries,
            "target": transcript
        }

    def __len__(self):
        # nb de fois où getitem est appelée dans une epoch
        return 180

    def crop_to_max_size(self, wav, target_size):
        return wav, 0

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = ([s["source"] for s in samples]) # = audios
        sources = torch.stack(sources).squeeze()
        padding_mask = ([s["padding_mask"] for s in samples])
        padding_mask = torch.stack(padding_mask).squeeze()
        boundaries = [s["boundaries"] for s in samples]
        ids = torch.LongTensor([s["id"] for s in samples])
        target = [s["target"] for s in samples]

        target_lengths = torch.LongTensor([len(t) for t in target])
        target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        ntokens = target_lengths.sum().item()

        net_input = {
            "source": sources,
            "boundaries": boundaries,
            "padding_mask": padding_mask
        }
        batch = {
            "id": ids,
            "net_input": net_input,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens
        }

        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sample_rate * self.max_sentence_length

    def ordered_indices(self):
        return np.random.permutation(len(self))