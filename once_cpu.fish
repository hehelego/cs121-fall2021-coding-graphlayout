#!/usr/bin/fish
set in_edge $argv[1]
set bench_file $argv[2]
set out_coor data/coordinate
set out_img data/image.png

set iter 100
set raw_size (python src/graph_size.py $in_edge)
set N (echo $raw_size | cut -d' ' -f1)
set M (echo $raw_size | cut -d' ' -f2)

echo -n "$in_edge " >> $bench_file
./bin/cpu $N $M $iter $in_edge $out_coor >> $bench_file
# python src/visualize.py $in_edge $out_coor $out_img
