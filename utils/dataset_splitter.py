import argparse
import os
import random
import shutil
import sklearn.model_selection

parser = argparse.ArgumentParser(description='utility script to split a NYUDepthV2 dataset to train, val, test')
parser.add_argument('--dataset_path', help='path to semantic image')
parser.add_argument('--train_fpath', help='path to depth image')
parser.add_argument('--val_fpath', help='path to rgb image')
args = parser.parse_args()

def main():
    with open(os.path.join(args.dataset_path, 'index.txt'), 'r') as f:
        lines = f.readlines()
        file_list = []
        for i in range(0, len(lines)):
            depth_fname = lines[i].strip()
            file_list.append(depth_fname)

        train, test = sklearn.model_selection.train_test_split(
            file_list, test_size=0.25, train_size=0.75)

        with open(args.train_fpath, "w") as train_f:
            for fpath in train:
                train_f.write("{}\n".format(fpath))

        with open(args.val_fpath, "w") as test_f:
            for fpath in test:
                test_f.write("{}\n".format(fpath))

if __name__ == '__main__':
    main()
