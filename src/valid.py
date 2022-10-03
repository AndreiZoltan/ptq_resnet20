import argparse
from model import load_model
from catalyst import dl
from dataset import CifarDS
from config import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.prog = "validate model on CIFAR10"

    parser.add_argument(
        "--path",
        default="{}/model.0047.pth".format(WEIGHTS_DIR),
        type=str,
        help="path to model",
    )
    return parser


def main(args):
    model = load_model(args.path)
    model.eval()

    dataset = CifarDS()
    valid_loader = dataset.get_valid_gen()

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
