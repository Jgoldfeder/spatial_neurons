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
# bash finetune_vit.sh $1 spatial 3000
# bash finetune_vit.sh $1 spatial 3500
# bash finetune_vit.sh $1 spatial 4000
# bash finetune_vit.sh $1 spatial 5500
# bash finetune_vit.sh $1 spatial 6000 
# bash finetune_vit.sh $1 spatial 6500 
# bash finetune_vit.sh $1 spatial 7000
# bash finetune_vit.sh $1 spatial 7500 
# bash finetune_vit.sh $1 spatial 8000 