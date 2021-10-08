from dataclasses import dataclass
import argparse
from typing import List

@dataclass
class ArgsModel:
    conf_path: str


def get_args_process(args: List[str]) -> ArgsModel:
    """Defines and parses command line arguments in program launch

    Args:
        args (List[str], optional): List of arguments to parse

    Returns:
        ArgsModel: Command line arguments with given values
    """

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-c",
        "--conf-path",
        required=True,
        help="Path to program configuration file in YAML format",
        type=str,
    )
    parsed_args = ap.parse_args(args)

    args_model = ArgsModel(**vars(parsed_args))

    return args_model
