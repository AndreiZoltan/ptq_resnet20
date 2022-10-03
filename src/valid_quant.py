import torch
import argparse
from catalyst import dl
from dataset import CifarDS
from glob import glob
import os
from config import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.prog = "validate model on CIFAR10"
    latest_model = max(glob("{}/*".format(QWEIGHTS_DIR)), key=os.path.getctime)
    parser.add_argument("--path", default=latest_model, type=str, help="path to model")
    parser.add_argument("-n", default=40, type=int, help="number of batches to valid")
    return parser


def main(args):
    model = torch.jit.load(args.path)
    model.eval()

    dataset = CifarDS()
    valid_loader = dataset.get_valid_gen(args.n)

    runner = dl.SupervisedRunner(
        input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    )

    runner.evaluate_loader(
        model=model,
        loader=valid_loader,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3)),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets"
            ),
        ],
        verbose=True,
    )


if __name__ == "__main__":
    argparser = build_parser()
    main(argparser.parse_args())
