#!/bin/bash

# device
bash finetune_vit.sh $1 spatial 20
bash finetune_vit.sh $1 spatial 40
bash finetune_vit.sh $1 spatial 80
bash finetune_vit.sh $1 spatial 120
bash finetune_vit.sh $1 spatial 162 
bash finetune_vit.sh $1 spatial 325 
bash finetune_vit.sh $1 spatial 650
bash finetune_vit.sh $1 spatial 1300 
bash finetune_vit.sh $1 spatial 2600 