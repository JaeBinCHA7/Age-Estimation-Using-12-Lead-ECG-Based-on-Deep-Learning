# Title: Age Estimation Using 12-Lead ECG Based on Deep Learning

## Introduction:
This project participated in the MAIC ECG AI Challenge 2023, which was centered around the development of age prediction models using electrocardiogram (ECG) data. The project achieved impressive results, solely relying on a deep learning network without any additional data or preprocessing.

## Project Structure:

* main

    options.py: Experimental environment settings.
    train_inferface.py: Deep learning model training.
    test_interface.py: Performance inference of the trained model.
* dataloader

    dataloader.py: Generation of data csv files and data loader.
* dataset

    train_dataset_adult.csv: Training data for adults.
    train_dataset_child.csv: Training data for children.
    valid_dataset_adult.csv: Validation data for adults.
    valid_dataset_child.csv: Validation data for children.
    Note: The provided data was divided into training and validation at a 9:1 ratio.
* log

    ResDenseNet_0923_adult: Logs and weights for the adult age estimation model.
    ResUDenseNet_0922_child: Logs and weights for the child age estimation model.

* models

    ResDenseNet.py: Network for adult age estimation.
    ResUDenseNet.py: Network for child age estimation.

* score

    Outputs from test_interface.py are saved in the submission.csv file.

* utils

    model_init.py: Definition of the network, Trainer, and Loss function.
    progress.py: Functions for training, data loading related functions.
    tensorboard.py: TensorBoard-related functions.
    trainer: Training and validation trainer.

# Usage
    
* Setting up option.py

  Configure the data path in option.py.

* Running train_interface.py

  Execute train_interface.py to generate the required csv files. These csv files map the paths to .npy files with their corresponding target age values.
  
  If you wish to train:
  * python train_interface.py --mode='adult' --env='0923_adult' --arch='ResDenseNet'
  *  python train_interface.py --mode='child' --env='0923_child' --arch='ResUDenseNet'

* Executing test_interface.py

  In option.py, the path to the best-performing weight is already set.

  Run the command:
  * python test_interface.py