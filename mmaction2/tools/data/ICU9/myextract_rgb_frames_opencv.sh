#!/usr/bin/env bash

cd ../
python build_rawframes.py D:/mmaction2-main/tools/data/ICU9/videos/ D:/mmaction2-main/tools/data/ICU9/rawframes/ --task rgb --level 2 --ext avi --new-width 1280 --new-height 720 --use-opencv
echo "Genearte raw frames (RGB only)"

cd ICU9/
