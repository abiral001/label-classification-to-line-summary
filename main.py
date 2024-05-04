from argparse import ArgumentParser
import yaml
import sys
from modules.lcl import LabelClassificationLlama
sys.path.append('.')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default='./config/config.yml', help="Path to config file")
    parser.add_argument("--train", "-t", action="store_true", help="Train the model Requires --mode to be set")
    parser.add_argument("--mode", "-m", type=str, default="lc", help="Train either of the models - Label Classification (lc), or optimize both (both)")
    args = parser.parse_args()
    # reading config file
    with open(args.config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    lcl = LabelClassificationLlama(config)
    lcl.train(args.mode)
    # text = "Random Text"
    # print(lcl.run(text))
    