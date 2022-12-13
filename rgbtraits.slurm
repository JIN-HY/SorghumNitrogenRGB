#!/bin/bash

source activate pcv

for m in {0..9}
do
python rgbtraits.py $1 $m 
done
