#!/usr/bin/fish
set in_edge data/wikivote.in
set out_coor data/wikivote.coordinate
set out_img data/wikivote.png
set iter 100
set raw_size (python src/graph_size.py $in_edge)
set N (echo $raw_size | cut -d' ' -f1)
set M (echo $raw_size | cut -d' ' -f2)

./bin/cpu $graph_size $iter $in_edge $out_coor

python src/visualize.py $in_edge $out_coor $out_img
