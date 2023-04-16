#!/bin/bash

RUN_CMD="python -u scripts/train.py num_workers=8 port=29452 learning_rate=2e-4 batch_size=16 experiment_name=test model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[64,32,16] model.decoder.n_channels=[16,32,64] trainer.max_epochs=300 data=era5_hpx32_7var_6h_24h data.src_directory=/home/mercury2/karlbam/Data/DLWP/HPX32 data.dst_directory=/home/mercury3/karlbam data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=True data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam model.enable_healpixpad=True"

# Command to run model on CPU (useful for prototyping and verifying code)
#NUM_CPU=4
#srun -u --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpu_bind=sockets \
#     -c $(( ${NUM_CPU} )) \
#     bash -c "
#     export WORLD_RANK=\${SLURM_PROCID}
#     export HDF5_USE_FILE_LOCKING=True
#     export CUDA_VISIBLE_DEVICES=
#     export HYDRA_FULL_ERROR=1 
#     ${RUN_CMD}"
#exit

# Run configuration
NUM_GPU=2
NUM_CPU=16
GPU_NAME=V100
DEVICE_NUMBERS="4,5"

# Command to run model on 
srun -u --ntasks=${NUM_GPU} \
     --ntasks-per-node=${NUM_GPU} \
     --gres=gpu:${GPU_NAME}:${NUM_GPU} \
     --cpu_bind=sockets \
     -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
     bash -c "
     export WORLD_RANK=\${SLURM_PROCID}
     export HDF5_USE_FILE_LOCKING=False
     export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
     export HYDRA_FULL_ERROR=1 
     ${RUN_CMD}"
