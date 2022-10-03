import torch
import torch.quantization.quantize_fx as quantize_fx
from datetime import datetime
import argparse
from model import *
from catalyst import dl
from dataset import CifarDS
from config import *


def build_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        default="{}/model.0047.pth".format(WEIGHTS_DIR),
        help="path tp model to quantize",
    )
    return argparser


def calibrate(model):
    dataset = CifarDS()
    train_loader = dataset.get_train_gen(1)
    runner = dl.SupervisedRunner(
        input_key="img", output_key="logits", target_key="targets", loss_key="loss"
    )
    runner.evaluate_loader(model=model, loader=train_loader)


def main(args):
    model = load_model(args.path)
    model.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
    model = quantize_fx.prepare_fx(model, qconfig_dict)
    calibrate(model)
    model = quantize_fx.convert_fx(model)

    model_name = "{}/qmodel_{}{}".format(
        QWEIGHTS_DIR, datetime.now().strftime("%H_%M_%S"), ".pth"
    )
    torch.jit.save(torch.jit.script(model), model_name)
    # torch.save(model.state_dict(), model_name)

    print("{} was saved".format(model_name))


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
