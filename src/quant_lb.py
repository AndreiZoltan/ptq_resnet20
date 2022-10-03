from qmodel import qresnet20
from model import load_model
import argparse
from catalyst import dl

# from config import *
from fusion import *


def build_parser():
    argparser = argparse.ArgumentParser()
    argparser.prog = "validate model on CIFAR10"

    argparser.add_argument(
        "--path",
        default="./weights/model.0047.pth",
        type=str,
        help="path to model",
    )
    return argparser


def fit(model: nn.Module, n_batches: int) -> None:
    from dataset import CifarDS

    dataset = CifarDS()
    train_loader = dataset.get_train_gen(n_batches)

    runner = dl.SupervisedRunner(
        input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    )

    runner.evaluate_loader(
        model=model,
        loader=train_loader,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3)),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets"
            ),
        ],
        verbose=True,
    )


def main(args):
    # init
    model = load_model(args.path).eval()
    qmodel = qresnet20(8)
    # prepare
    fuse_conv_bn(model)
    prepare2(model, qmodel)
    # calibrate
    fit(qmodel, 2)
    # quantize
    qmodel.quantize()
    # evaluate
    fit(qmodel, 5)


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
