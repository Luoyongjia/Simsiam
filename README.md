# Simsiam
This is a pyTorch re-implementation of [SimSiam](https://arxiv.org/pdf/2011.10566.pdf).
## Requirements
pytorch==1.3.1,   
torchvision==0.4.2,     
tqdm,     
matplotlib.
## Datasets
Please change the datasets dir in `Utils/parser.py`.
## Experiment
|learning rate|batch_size|Encoder|Acc (linear eval)|
|-----|-----|-----|-----|
|0.16|512|ResNet18|91.66|t18