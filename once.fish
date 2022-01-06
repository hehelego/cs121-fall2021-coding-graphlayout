#!/usr/bin/fish
set in_edge data/wiki-vote.txt
set out_coor data/wiki-vote.coordinate
set out_img data/wiki-vote.png
set iter 100
set graph_size (python src/graph_size.py $in_edge)

bin/cpu $graph_size $iter $in_edge $out_coor
python src/visualize.py $in_edge $out_coor $out_img
