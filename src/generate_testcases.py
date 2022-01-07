#!/usr/bin/python3

    # data/clique100 data/clique200 data/clique300 data/clique400 data/clique500 \
    # data/wikivote data/Gowalla

from typing import IO
from random import randint as rand

def gen_clique(n: int, f: IO[str]):
    for i in range(n):
        for j in range(n):
            if i<j:
                print(f'{i} {j}',file=f)
def gen_tree(n: int, f: IO[str]):
    for i in range(1,n):
        j = rand(0,i-1)
        print(f'{j} {i}',file=f)

if __name__ == '__main__':
    n = [100,200,300,400,500]

    for i in n:
        with open(f'data/clique{i}','w') as f:
            gen_clique(i,f)
        with open(f'data/tree{i}','w') as f:
            gen_tree(i,f)