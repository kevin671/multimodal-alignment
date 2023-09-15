# multimodal-alignment

## Implemented algorithms

- SimCLR
- Barlow Twins
- VICReg

## Set up

```
sudo singularity build container.sif container.def
```

Run the container:

```
singularity run --nv container.sif 
```

Run a command within a container:
```
singularity exec --nv container.sif 'my-command'
```

## Training

Due to the requirement of a large batchsize, we highly recommend you to use DDP training. A slurm-based script is as below:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -u toData Preparationols/train.py \
    --name kit_baseline_dp_2gpu_8layers_1000 \
    --batch_size 128 \
```

Otherwise, you can run the training code on a single GPU like:

```shell
python -u tools/train.py \
    --name kit_baseline_1gpu_8layers_1000 \
    --batch_size 128 \
```

## Acknowledgement

This code is developed on top of:

- [SimCLR](https://github.com/sthalles/SimCLR)
- [Barlow Twins and HSIC](https://github.com/yaohungt/Barlow-Twins-HSIC)
- [VICReg](https://github.com/facebookresearch/vicreg)
