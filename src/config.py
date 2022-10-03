from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = "{}/logdir".format(PARENT_DIR)
WEIGHTS_DIR = "{}/weights".format(PARENT_DIR)
TENSORBOARD_DIR = "{}/tensorboard".format(LOG_DIR)
QWEIGHTS_DIR = "{}/qweights".format(PARENT_DIR)

# __all__ = [PARENT_DIR, LOG_DIR, WEIGHTS_DIR, TENSORBOARD_DIR]
