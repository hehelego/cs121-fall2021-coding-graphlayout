#!/usr/bin/python

from typing import Tuple
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def parse_args() -> Tuple[str, str, str]:
    this_file = sys.argv[0]
    usage_text = [
        f'a python script to visualize the layout',
        f'python {this_file} input_snap input_coordinate output',
        '\tinput_snap(required): path',
        '\t\tPath to the edge file',
        '\tinput_coordinate(required): path',
        '\t\tPath to the coordinate file',
        '\t\tA triple of space separated real number on each line: x y-coordinate centrality',
        '\toutput(required): path',
        '\t\tThe output image will be saved in the output file',
    ]

    parser = argparse.ArgumentParser()
    parser.usage = '\n'.join(usage_text)
    parser.add_argument('input_snap', type=str, help='path to the input file')
    parser.add_argument('input_coordinate', type=str,
                        help='path to the input file')
    parser.add_argument('output', type=str,
                        help='path where the output is saved')
    args = parser.parse_args(sys.argv[1:])
    ie, ic, o = args.input_snap, args.input_coordinate, args.output
    assert os.path.isfile(ie)
    assert os.path.isfile(ic)
    return str(ie), str(ic), str(o)


def main():
    path_edge, path_coordinate, path_out = parse_args()

    raw_edge = np.loadtxt(path_edge, delimiter=' ').astype(int)
    raw_coordinate = np.loadtxt(path_coordinate, delimiter=' ')
    xs, ys, ws = raw_coordinate.transpose()
    points = raw_coordinate[:, :-1]

    plt.scatter(x=xs, y=ys, s=ws*100, color='red')
    plt.gca().add_collection(LineCollection(points[raw_edge]))
    plt.show()
    plt.savefig(path_out)


if __name__ == '__main__':
    main()