#!/bin/bash
#  Tange, O. (2021, September 22). GNU Parallel 20210922 ('Vindelev').
#  Zenodo. https://doi.org/10.5281/zenodo.5523272
# run program X times consecutively
#seq 6 | xargs -I{} z python3 main.py
seq 1 6 | parallel -j 6 -I{} python3 main.py