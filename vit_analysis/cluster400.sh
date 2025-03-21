#!/bin/bash

# device
bash finetune_vit.sh $1 cluster400 20
bash finetune_vit.sh $1 cluster400 40
bash finetune_vit.sh $1 cluster400 80
bash finetune_vit.sh $1 cluster400 120
bash finetune_vit.sh $1 cluster400 162 
bash finetune_vit.sh $1 cluster400 325 
bash finetune_vit.sh $1 cluster400 650
bash finetune_vit.sh $1 cluster400 1300 
bash finetune_vit.sh $1 cluster400 2600 

bash finetune_vit.sh $1 cluster400 3000
bash finetune_vit.sh $1 cluster400 3500
bash finetune_vit.sh $1 cluster400 4000
bash finetune_vit.sh $1 cluster400 5500
bash finetune_vit.sh $1 cluster400 6000 
bash finetune_vit.sh $1 cluster400 6500 
bash finetune_vit.sh $1 cluster400 7000
bash finetune_vit.sh $1 cluster400 7500 
bash finetune_vit.sh $1 cluster400 8000 