# multimodal-alignment

## Contrastive Multiview Coding (Tian et al., 2019)

Training CMC with ResNets requires at least 4 GPUs, the command of using resnet50v1 looks like:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python CMC_train.py --model resnet50v1 --batch_size 128 --num_workers 24
 --data_folder ../data/ \
 --model_path ./checkpoints \
 --tb_path ./tensorboard \
```

Training Linear Classifier:

```shell
CUDA_VISIBLE_DEVICES=0 python CMC_LinearProbing.py --dataset imagenet \
 --data_folder /path/to/data \
 --save_path /path/to/save \
 --tb_path /path/to/tensorboard \
 --model_path ./models/alexnet.pth \
 --model resnet50v1 --learning_rate 30 --layer 6 
```

Train and evaluate on ILSVRC 2012: `Acc@1 48.200 Acc@5 74.156`

## Acknowledgement

This code is developed on top of:

- [SimCLR](https://github.com/sthalles/SimCLR)
- [Barlow Twins and HSIC](https://github.com/yaohungt/Barlow-Twins-HSIC)
- [VICReg](https://github.com/facebookresearch/vicreg)
