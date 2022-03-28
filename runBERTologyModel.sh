#!/bin/bash


for model in bert-base-uncased
    do
    mkdir -p ./Records/${model}
    
    for seed in 1 2 3
        do
        nohup python -u BERTologyModel.py --PretrainModel ${model} --RecordsDir ./Records/${model} > ./Records/${model}/english2others${seed}.txt 2>&1
        done
    done
