# StableVQA

可复现论文["StableVQA: A Deep No-Reference Quality Assessment Model for Video Stability"](https://arxiv.org/abs/2308.04904)

## Database

抖动视频数据集下载[StableDB](https://drive.google.com/file/d/1XO1tkmSNg-yPcfQ0WSnpvB3mu0bILZQA/view?usp=drive_link)

## Training

    python new_train.py -o ./options/stable.yml

## Testing

    python new_test.py -o ./options/stable.yml
