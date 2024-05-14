import argparse
import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flame-Video-Fitting")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    args = parser.parse_args()

    conf = util.conf.read_conf(args.conf)