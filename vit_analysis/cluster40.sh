#!/bin/bash

# device
bash finetune_vit.sh $1 cluster40 20
bash finetune_vit.sh $1 cluster40 40
bash finetune_vit.sh $1 cluster40 80
bash finetune_vit.sh $1 cluster40 120
bash finetune_vit.sh $1 cluster40 162 
bash finetune_vit.sh $1 cluster40 325 
bash finetune_vit.sh $1 cluster40 650
bash finetune_vit.sh $1 cluster40 1300 
bash finetune_vit.sh $1 cluster40 2600 

bash finetune_vit.sh $1 cluster40 3000
bash finetune_vit.sh $1 cluster40 3500
bash finetune_vit.sh $1 cluster40 4000
bash finetune_vit.sh $1 cluster40 5500
bash finetune_vit.sh $1 cluster40 6000 
bash finetune_vit.sh $1 cluster40 6500 
bash finetune_vit.sh $1 cluster40 7000
bash finetune_vit.sh $1 cluster40 7500 
bash finetune_vit.sh $1 cluster40 8000 