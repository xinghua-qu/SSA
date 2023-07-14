#!/bin/bash

pip3 install -r requirements.txt
cd deepspeech.pytorch
pip3 install -e .
cd ..
pip3 install git+https://github.com/pytorch/hydra-torch
pip3 install git+https://github.com/romesco/hydra-lightning/#subdirectory=hydra-configs-pytorch-lightning

# Download the pretrained VITS model from the link below
# https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2
# Also down the corresponding datasets, such as Common Voice and LibriSpeech
