from yacs.config import CfgNode as CN

_C = CN()

_C.TEXT_PATH = "/data/49010-0.txt"
_C.BATCH_SIZE = 32
_C.BPTT_LEN = 64
_C.NUM_EPOCHS = 50

_C.N_HIDDEN = 2048
_C.N_LAYERS = 1
_C.LR = 0.001

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()
