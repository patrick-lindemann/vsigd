#!/bin/bash

# env: vsigd
pip install --upgrade pip && \
    conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia && \
    conda install -y suitesparse==5.10.1 scikit-sparse==0.4.8 -c conda-forge && \
    conda install -y pyg pytorch-scatter pytorch-cluster -c pyg && \
    pip install -r requirements.txt && \
    conda install -y pytorch3d -c pytorch3d
