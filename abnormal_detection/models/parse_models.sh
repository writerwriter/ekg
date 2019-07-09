#!/usr/bin/env bash

for filename; do
    wget "https://api.wandb.ai/toosyou/ekg-abnormal_detection/$filename/model-best.h5" -O "$filename.h5" &
done
