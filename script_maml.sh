#! /bin/bash

METHOD=maml
TASK=cosine

BATCHSIZE=5
DATASIZE=5

STEPSIZE=0.001
NSTEP=1
NTRAIN=1000000

python run.py \
    --task $TASK \
    --method $METHOD \
    --batchsize $BATCHSIZE \
    --datasize $DATASIZE \
    --stepsize $STEPSIZE \
    --nstep $NSTEP \
    --ntrain $NTRAIN \
    > log/log\_maml\_$TASK\_$BATCHSIZE\_$DATASIZE\_$NSTEP.txt \
    2> log/err_maml\_$TASK\_$BATCHSIZE\_$DATASIZE\_$NSTEP.txt &



