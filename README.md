# StableVQA

Code for paper ["StableVQA: A Deep No-Reference Quality Assessment Model for Video Stability"](https://arxiv.org/abs/2308.04904)

## Database

Download the [StableDB](https://drive.google.com/file/d/1XO1tkmSNg-yPcfQ0WSnpvB3mu0bILZQA/view?usp=drive_link), including 1952 unstable videos with corresponding MOSs.

## Pretrained Weights

Download RAFT pre-training weight [raft-things.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

Download StripFormer pre-training weight [Stripformer_realblur_J.pth](https://drive.google.com/drive/folders/1YcIwqlgWQw_dhy_h0fqZlnKGptq1eVjZ?usp=sharing)

Download Swin Transformer pre-training weight [swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

The default path for these pre-training weights is `./pretrained_weights/*.pth`.

## Training

    python new_train.py -o ./options/stable.yml

## Testing

    python new_test.py -o ./options/stable.yml
