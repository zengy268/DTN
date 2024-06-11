# Disentanglement Translation Network
This is the open source code for paper: Disentanglement Translation Network for Multimodal Sentiment Analysis

I have just re-organized the codes but haven't finished proofreading yet. Please feel free to contact me by email.

## Table of Contents
- [Description](##Description)
- [Preparation](##Preparation)
- [Running](##Running)

## Description
This is the open source code for paper: Multimodal Translation Network for Multimodal Sentiment Analysis. We have provided the implementation on the task of multimodal sentiment analysis with the following main components:
1. `./datasets` contains the datasets used in the experiments
2. `./modules` contains the model definition
3. `./utils` contains the functions for data processing, evaluation metrics, etc.
4. `global_configs.py` defines important constants
5. `train.py` defines the training process

## Preparation
### Datasets
The processed MOSI and MOSEI datasets will be downloaded in `./dataset' by running `datasets/download_datasets.sh`

### Configuration
Before starting training, please define the global constants in `global_configs.py`. Default configuration is set to MOSI dataset. Important settings include GPU setting, learning rate, feature dimension, training epochs, training batch size, dataset setting and dimension setting of input data. To run the MOSEI dataset, remember to change dimension of the visual modality

```
from torch import nn

class DefaultConfigs(object):

    device = '1'                                 #GPU setting
    logs = './logs/'
    
    max_seq_length = 50 
    lr = 1e-5                                    #learning rate
    d_l = 80                                     #feature dimension
    n_epochs = 100                               #training epochs
    train_batch_size = 16                        #training batch size
    dev_batch_size = 128
    test_batch_size = 128
    model_choice = 'bert-base-uncased'

    dataset = 'mosi'                             #dataset setting
    TEXT_DIM = 768                               #dimension setting
    ACOUSTIC_DIM = 74
    VISUAL_DIM = 47

    # dataset = 'mosei'
    # ACOUSTIC_DIM = 74
    # VISUAL_DIM = 35
    # TEXT_DIM = 768

config = DefaultConfigs()
```

## Running
To run the experiments, please run
```
python train.py
```

## Acknowledgments
We would like to express our gratitude to [huggingface](https://huggingface.co/) and [MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer), which are of great help to our work.
