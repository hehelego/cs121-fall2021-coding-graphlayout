#!/usr/bin/python

import sys
import os
import numpy as np


def print_usage():
    usage = [
        'utility to get the vertices and edges count of a SANP-formate graph'
        f'python {sys.argv[0]} input',
        '\ttinput: path to a SNAP graph file',
    ]
    print(*usage, sep='\n')


def parse_args() -> str:
    if len(sys.argv) < 2:
        print_usage()
        exit(1)
    path = sys.argv[1]
    assert os.path.isfile(path)
    return path


def main():
    graph_path = parse_args()
    edge_list = np.loadtxt(graph_path).astype(int)

    vertices = edge_list.max()+1
    edges = edge_list.shape[0]
    print(vertices, edges)


if __name__ == '__main__':
    main()
