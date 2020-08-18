#! /bin/bash

METHOD=gml
TASK=cosine

BATCHSIZE=5
DATASIZE=5

STEPSIZE=0.001
NSTEP=1
NTRAIN=1000000

NU=2

SAMPLING=1
NPARAMSAMPLE=5

ADAPTIVEPIVOT=1
NPIVOT=100

python run.py \
    --task $TASK \
    --method $METHOD \
    --batchsize $BATCHSIZE \
    --datasize $DATASIZE \
    --stepsize $STEPSIZE \
    --nstep $NSTEP \
    --ntrain $NTRAIN \
    --nu $NU \
    --sampling $SAMPLING \
    --nparamsample $NPARAMSAMPLE \
    --adaptivepivot $ADAPTIVEPIVOT \
    --npivot $NPIVOT \
    > log/log\_gml\_$TASK\_$BATCHSIZE\_$DATASIZE\_$NSTEP\_$NU\_$SAMPLING\_$ADAPTIVEPIVOT.txt \
    2> log/err_gml\_$TASK\_$BATCHSIZE\_$DATASIZE\_$NSTEP\_$NU\_$SAMPLING\_$ADAPTIVEPIVOT.txt &



