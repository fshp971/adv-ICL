import pickle, os, sys, logging, yaml
import torch
from safetensors.torch import safe_open, save_file


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info("Arguments")
    for arg in vars(args):
        logger.info("    {:<22}        {}".format(arg+":", getattr(args,arg)) )
    logger.info("")

    return logger


def load_yaml(path: str = None, dump_path: str = None):
    res = None
    if path is not None:
        with open(path, "r") as f:
            res = yaml.load(f, Loader=yaml.loader.SafeLoader)
        save_yaml(res, dump_path)
    return res


def save_yaml(config: dict, path: str = None):
    if path is not None:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def load_state_dict(path: str, mode: str = "pytorch"):
    if mode == "pytorch":
        state_dict = torch.load(path, map_location="cpu")

    elif mode == "safetensors":
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    else:
        raise RuntimeError(f"Unknown loading mode `{mode}`")

    return state_dict


def save_state_dict(state_dict: dict, path: str, mode: str = "pytorch"):
    if mode == "pytorch":
        torch.save(state_dict, path)

    elif mode == "safetensors":
        save_file(state_dict, path)

    else:
        raise RuntimeError(f"Unknown saving mode `{mode}`")
