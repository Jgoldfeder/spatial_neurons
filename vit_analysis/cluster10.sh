#!/bin/bash

# device
bash finetune_vit.sh $1 cluster10 20
bash finetune_vit.sh $1 cluster10 40
bash finetune_vit.sh $1 cluster10 80
bash finetune_vit.sh $1 cluster10 120
bash finetune_vit.sh $1 cluster10 162 
bash finetune_vit.sh $1 cluster10 325 
bash finetune_vit.sh $1 cluster10 650
bash finetune_vit.sh $1 cluster10 1300 
bash finetune_vit.sh $1 cluster10 2600 

bash finetune_vit.sh $1 cluster10 3000
bash finetune_vit.sh $1 cluster10 3500
bash finetune_vit.sh $1 cluster10 4000
bash finetune_vit.sh $1 cluster10 5500
bash finetune_vit.sh $1 cluster10 6000 
bash finetune_vit.sh $1 cluster10 6500 
bash finetune_vit.sh $1 cluster10 7000
bash finetune_vit.sh $1 cluster10 7500 
bash finetune_vit.sh $1 cluster10 8000 
