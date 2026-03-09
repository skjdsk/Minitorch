#!/usr/bin/env bash

if [ -d data ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir data
cd data

# GitHub mirror URLs
BASE_URL="https://raw.githubusercontent.com/fgnt/mnist/master"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
    echo "Downloading $f ..."
    curl -O "$BASE_URL/$f"
done

echo "Unzipping..."
gunzip *.gz

echo "Done."