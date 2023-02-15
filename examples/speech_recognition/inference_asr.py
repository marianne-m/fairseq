import argparse
import logging
import torch
from typing import List
from os import makedirs
from pathlib import Path
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import Dictionary
import fairseq.meters
import yaml
# from evaluate import load
from jiwer import wer

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_blanks(phrase: List[str], tgt_dict: Dictionary):
    symb_to_remove = [
        tgt_dict[tgt_dict.bos()],
        tgt_dict[tgt_dict.pad()],
        tgt_dict[tgt_dict.eos()],
        tgt_dict[tgt_dict.unk()]
    ]
    clean_phrase = list()
    for symb in phrase:
        if symb not in symb_to_remove:
            clean_phrase.append(symb)
    return clean_phrase


def remove_duplicates(phrase: List[str]):
    prev_symb = None
    clean_phrase = list()
    for symb in phrase:
        if symb != prev_symb:
            clean_phrase.append(symb)
        prev_symb = symb
    return clean_phrase


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=700000,
        # max_sentences=args.batch_size,
        # max_positions=(sys.maxsize, sys.maxsize),
        # ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        # required_batch_size_multiple=args.required_batch_size_multiple,
        # num_shards=args.num_shards,
        # shard_id=args.shard_id,
        # num_workers=args.num_workers,
        # data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None,
            # need_attn=args.print_alignment,
        )
        # if args.fp16:
        #     model.half()
        if use_cuda:
            model.cuda()


def main(args):

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.model]
    )
    use_cuda = torch.cuda.is_available()
    optimize_models(args, use_cuda, models)

    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    # Set dictionary
    tgt_dict = task.target_dictionary

    logger.info(
        "| {} {} {} examples".format(
            args.data, args.gen_subset, len(task.dataset(args.gen_subset))
        )
    )

    itr = get_dataset_itr(args, task, models)

    ground_truth = list()
    hypothesis = list()

    for sample in itr:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        target_phrase = [tgt_dict[i] for i in sample['target'][0]]

        encoder_out = models[0](**sample["net_input"])

        symbols = list()
        for frame in encoder_out["encoder_out"]:
            argmax = torch.argmax(frame[0])
            symbols.append(tgt_dict[argmax])

        inferred_phrase = remove_duplicates(symbols)
        inferred_phrase = remove_blanks(inferred_phrase, tgt_dict)

        target_phrase = "".join(target_phrase).replace("|", " ")
        inferred_phrase = "".join(inferred_phrase).replace("|", " ")

        ground_truth.append(target_phrase)
        hypothesis.append(inferred_phrase)

    # wer = jiwer.wer(ground_truth, hypothesis)
    wer_score = wer(ground_truth, hypothesis)
    logger.info(
        f"| WER = {wer_score}"
    )

    makedirs(args.results_path, exist_ok=True)
    with open(args.results_path / "wer.yml", "w") as file:
        yaml.dump({"wer": wer_score}, file)    



def parser():
    parser = argparse.ArgumentParser(description="Do inference and compute WER")
    parser.add_argument("data", type=Path,
                        help="Path to data")
    parser.add_argument("--model", required=True, type=str,
                        help="Model checkpoint to use for inference")
    parser.add_argument("--gen-subset", required=True, type=str,
                        help="Subset to test")
    parser.add_argument("--results-path", required=True, type=Path,
                        help="Path to store WER score")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    main(args)