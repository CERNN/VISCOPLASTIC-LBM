import copy
from typing import Any, List, Dict, Callable, TypeVar, Type
from ruamel.yaml import YAML

from pyvisc.schemes import SchemeVarH, SchemeIBMVarH
T = TypeVar("T")

def _dict_to_configs(cfg_dict: Dict, scheme: Type[T]) -> T:
        cfg_cp = copy.deepcopy(cfg_dict)
        scheme_inst = scheme(**cfg_cp ) # type: ignore
        return scheme_inst

def read_configs(filename: str) -> SchemeVarH:
    try:
        # Read YAML from file
        with open(filename, "r") as f:
            yaml = YAML(typ="safe")
            cfg_dicts = yaml.load(f)
            cfg = _dict_to_configs(cfg_dicts[0], SchemeVarH)
            return cfg
    except Exception as e:
        raise ValueError(f"Unable to load YAML from {filename}. Exception {e}") from e
