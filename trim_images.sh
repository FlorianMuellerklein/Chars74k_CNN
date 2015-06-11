#!/bin/bash

IMAGES="/Volumes/Mildred/Kaggle/chars_74k/data/test/*.Bmp"
for file in $IMAGES
do
	echo "$file"
	convert $file -resize 64x64! -gravity center $file

done
