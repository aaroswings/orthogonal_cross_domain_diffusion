from trainer.UtilFunctions import resize_reformat_files

import argparse

parser = argparse.ArgumentParser(
    prog='ResizeFiles'
)

parser.add_argument('dir_in', type=str)
parser.add_argument('dir_out', type=str)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--do_crop_by_ratio', type=bool, default=False)

args = parser.parse_args()

if __name__ == '__main__':
    resize_reformat_files(args.dir_in, args.dir_out, args.resolution, args.do_crop_by_ratio) 