#!/bin/bash

module load anaconda/2020.11
module load cuda/12.1
source activate climsim

python train_new.py --n_fold 0