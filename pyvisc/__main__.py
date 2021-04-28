import sys

from pyvisc.args_handler import get_args_process
from pyvisc.conf_handler import read_configs

def main(*args):
    args_p = get_args_process(args)
    cfg = read_configs(args_p.conf_path)
    cfg.render("pyvisc/generated")



if(__name__== "__main__"):
    main(*sys.argv[1:])