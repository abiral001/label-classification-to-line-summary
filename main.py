from argparse import ArgumentParser
import yaml
import sys
import os
import json
from modules.lcl import LabelClassificationLlama
sys.path.append('.')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default='./config/config.yml', help="Path to config file")
    parser.add_argument("--train", "-t", action="store_true", help="Train the model Requires --mode to be set")
    parser.add_argument("--mode", "-m", type=str, default="both", help="Train either of the models - Label Classification (lc) or Use Inference (infer)")
    parser.add_argument("--input_file", "-i", type=str, default=None, help="Input file path")
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    lcl = LabelClassificationLlama(config)
    if args.train:
        lcl.train(args.mode)
    else:
        with open(args.input_file, 'r') as f:
            text = f.readlines()
        text = "".join(text).replace('\n', ' ')
        output = lcl.run(text)
        with open("./output.txt", "w") as f:
            f.write(output)
        print("Summary output written to ./output.txt")
    