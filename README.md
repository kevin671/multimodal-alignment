# multimodal-alignment

## Set up

```shell
sudo singularity build mm_container.sif mm_container.def
```

Run the container:

```
singularity exec container.sif 'my-command'
```

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

Train and evaluate on ILSVRC 2012: `Acc@1 56.104 Acc@5 79.316`

## Barlow Twins (Zbontar et al., 2021) and HSIC

This code is developed on top of [Barlow-Twins-HSIC](https://github.com/yaohungt/Barlow-Twins-HSIC/tree/main).

### Supported Dataset
`CIFAR10`, `STL10`, and [`Tiny_ImageNet`](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4).

### Train and Linear Evaluation using Barlow Twins 
```shell
# ACC@1: 91.32% ACC@5: 99.81%
python Barlow_Twins_HSIC_main.py --lmbda 0.0078125 --corr_zero --batch_size 128 --feature_dim 128 --dataset cifar10
python Barlow_Twins_HSIC_linear.py --dataset cifar10 --model_path results/0.0078125_128_128_cifar10_model.pth
```

### Train and Linear Evaluation using HSIC
```shell
# ACC@1: 91.44% ACC@5: 99.77%
python Barlow_Twins_HSIC_main.py --lmbda 0.0078125 --corr_neg_one --batch_size 128 --feature_dim 128 --dataset cifar10
python Barlow_Twins_HSIC_linear.py --dataset cifar10 --model_path results/neg_corr_0.0078125_128_128_cifar10_model.pth
```

## Acknowledgement

This code is developed on top of
