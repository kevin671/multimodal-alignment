# multimodal-alignment

## Implemented algorithms

- SimCLR
- Barlow Twins
- VICReg

## Set up


## Training

Training CMC with ResNets requires at least 4 GPUs, the command of using resnet50v1 looks like:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_CMC.py --model resnet50v1 --batch_size 128 --num_workers 24
 --data_folder ../data/ \
 --model_path ./checkpoints \
 --tb_path ./tensorboard \
```

## Acknowledgement

This code is developed on top of:

- [SimCLR](https://github.com/sthalles/SimCLR)
- [Barlow Twins and HSIC](https://github.com/yaohungt/Barlow-Twins-HSIC)
- [VICReg](https://github.com/facebookresearch/vicreg)
