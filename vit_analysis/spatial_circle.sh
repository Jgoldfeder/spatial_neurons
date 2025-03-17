#!/bin/bash

# device
bash finetune_vit.sh $1 spatial-circle 20
bash finetune_vit.sh $1 spatial-circle 40
bash finetune_vit.sh $1 spatial-circle 80
bash finetune_vit.sh $1 spatial-circle 120
bash finetune_vit.sh $1 spatial-circle 162 
bash finetune_vit.sh $1 spatial-circle 325 
bash finetune_vit.sh $1 spatial-circle 650
bash finetune_vit.sh $1 spatial-circle 1300 
bash finetune_vit.sh $1 spatial-circle 2600 
