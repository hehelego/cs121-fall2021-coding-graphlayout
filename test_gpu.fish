#!/usr/bin/fish
set in_edge data/test
set out_coor data/coordinate
set out_img data/image.png

set iter 100
set raw_size (python src/graph_size.py $in_edge)
set N (echo $raw_size | cut -d' ' -f1)
set M (echo $raw_size | cut -d' ' -f2)

set -x CUDA_BLOCK_X 128
set -x CUDA_BLOCK_Y 8


./bin/gpu $N $M $iter $in_edge $out_coor
python src/visualize.py $in_edge $out_coor $out_img
