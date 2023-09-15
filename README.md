# multimodal-alignment

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
    --times 50 \
    --num_epochs 50 \
    --dataset_name kit \
    --num_layers 8 \
    --diffusion_steps 1000 \
    --data_parallel \
    --gpu_id 0 1
```

Otherwise, you can run the training code on a single GPU like:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/train.py \
    --name kit_baseline_1gpu_8layers_1000 \
    --batch_size 128 \
    --times 25 \
    --num_epochs 50 \
    --dataset_name kit
```

Here, `times` means the duplication times of the original dataset. To retain the number of iterations, you can set `times` to 25 for 1 GPU, 50 for 2 GPUs, 100 for 4 GPUs, and 200 for 8 GPUs.

