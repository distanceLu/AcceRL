# src/utils.py
import yaml
import argparse
from copy import deepcopy
from types import SimpleNamespace


def load_config(path, mode="defaults"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    base = deepcopy(raw["defaults"])
    if mode != "defaults" and mode in raw:
        base = merge_dicts(base, raw[mode])

    return base


def merge_dicts(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def update_config(cfg, overrides):
    """
    Update nested dict `cfg` using dot-notation keys like 'training.lr'
    """
    for key, value in overrides.items():
        parts = key.split(".")
        sub = cfg
        for p in parts[:-1]:
            sub = sub.setdefault(p, {})
        # Try to cast numeric values
        try:
            value = eval(value)
        except:
            pass
        sub[parts[-1]] = value
    return cfg

def parse_config(args: argparse.Namespace) -> SimpleNamespace:
    cfg_dict = load_config(args.config, mode=args.mode)
    overrides = dict(item.split("=") for item in args.overrides)
    cfg_dict = update_config(cfg_dict, overrides)
    cfg = dict_to_namespace(cfg_dict)
    
    return cfg, cfg_dict
