#!/bin/bash

TEST_PHOTO=$1

if [ -z "$TEST_PHOTO" ]; then
    echo "Error: No test photo provided."
    echo "Usage: ./analyze.sh path/to/photo.jpg"
    exit 1
fi

echo "==================================================="
echo "[1/3] Conditioning reference data via DECA..."
echo "==================================================="
python DECA/demos/demo_reconstruct.py -i DECA/TestSamples/examples --savefolder DECA/TestSamples/examples/results --saveImages True

echo -e "\n==================================================="
echo "[2/3] Routing clean data to Training Folder..."
echo "==================================================="
python bridge.py

echo -e "\n==================================================="
echo "[3/3] Calculating Accuracy Score..."
echo "==================================================="
python recognize.py "$TEST_PHOTO"