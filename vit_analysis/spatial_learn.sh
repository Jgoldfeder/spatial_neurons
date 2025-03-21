#!/bin/bash

# device
bash finetune_vit.sh $1 spatial-learn 20
bash finetune_vit.sh $1 spatial-learn 40
bash finetune_vit.sh $1 spatial-learn 80
bash finetune_vit.sh $1 spatial-learn 120
bash finetune_vit.sh $1 spatial-learn 162 
bash finetune_vit.sh $1 spatial-learn 325 
bash finetune_vit.sh $1 spatial-learn 650
bash finetune_vit.sh $1 spatial-learn 1300 
bash finetune_vit.sh $1 spatial-learn 2600 
