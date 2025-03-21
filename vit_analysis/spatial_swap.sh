#!/bin/bash

# device
bash finetune_vit.sh $1 spatial-swap 20
bash finetune_vit.sh $1 spatial-swap 40
bash finetune_vit.sh $1 spatial-swap 80
bash finetune_vit.sh $1 spatial-swap 120
bash finetune_vit.sh $1 spatial-swap 162 
bash finetune_vit.sh $1 spatial-swap 325 
bash finetune_vit.sh $1 spatial-swap 650
bash finetune_vit.sh $1 spatial-swap 1300 
bash finetune_vit.sh $1 spatial-swap 2600 
