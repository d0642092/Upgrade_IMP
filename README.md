# Upgrade_IMP

* Original code author: 
    k-r-allen
    https://github.com/k-r-allen/imp
* Original author: 
Kelsey Allen, Evan Shelhamer, Hanul Shin, Josh Tenenbaum.
* Paper name:
Infinite Mixture Prototypes for Few-Shot Learning.

**This code is upgrade IMP from python 2.x to python 3.7, only implement mini-ImageNet.**

## Execute
```python run_eval.py --data-root=./DATA/img/ --dataset="mini-imagenet" --label-ratio 0.4 --num-unlabel-test=5 --num-unlabel=5 --nclasses-train=5 --nclasses-episode=5 --nclasses-eval=5 --model "imp" --results "Result/mini-imagenet/1_5/" --nshot=1 --seed=0```
### env setting
dataset="mini-imagenet"  #dataset name  
labelratio=0.4 #fraction of  labeled data  
numunlabel=5 #number of unlabeled examples  
nclassestrain=5 #number of classes to sample from   
nclassesepisode=5 #n-way of episode  
accumulationsteps=1 #number of gradient accumulation steps before propagating loss (refer to paper)  
nclasseseval=5 #n-way of test episodes  
nshot=1 #shot of episodes  
model="imp" #model name  
seed=0 #seed  
results="Result"  


## Folder(src):
* factory: pull all factory code
    * __init__.py: 'from src import factory' will run
    * config_factory.py: register all config
    * data_factory.py: register all data
    * model_factory.py: register all model
* configs: hyperparameter setting
    * __init__.py: 'from src import configs' will run
    * mini_imagenet_config.py: hyperparameter for mini-imagenet
* data: data setting
    * __init__.py: 'from src import data' will run
    * episode.py: data separate operation function
    * refinement_dataset.py: parent of all data class
    * mini_imagenet: data for mini-imagenet
        1. setting parameter than call parent init func to read data
        2. check data cache is no create new cache (image: .hdf5, label: .pkl)
* models:
    * __init__.py: 'from src import models' will run
    * basic.py: parent of all model
* utils
