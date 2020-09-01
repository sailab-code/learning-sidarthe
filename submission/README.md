# An Optimal Control Approach to Learning in SIDARTHE Epidemic model

In this README you will find the information on how to install the dependencies for running the code and how to run the training and evaluation code.

## Installation

The code was developed using python 3.7.6. Previous python version may not be able to run the code.

There are two ways of installing the dependencies for the code.
* through conda
* through pip

The quickest and recommended way should be installing with [__conda__](https://docs.conda.io/en/latest/). In case conda is not available or you have trouble with installation, you can use pip.


### 1. Install with conda

After installing conda, browse to the code root directory and run the following command:  

`conda env create -f environment.yml`

It will download several packages and create a conda environment called *covid-tools*. Switch to that conda environment with the command:  

`conda activate covid-tools`

At this point, the code should be ready to be executed.

### 2. Install with pip
To install with pip, you will need to manually install the following packages:

* pytorch==1.2.0
* tensorboard==1.15.0
* numpy==1.18.1
* pandas==1.0.3
* matplotlib==3.2.1

At this point, the code should be ready to be executed.

## Running the code

The code provided is ready to run. It comes with the official data for all italian regions and for Italy as a whole. The data is stored in the directory `COVID-19`.

The file `nature_results.csv` contains the predictions of the model from Giordano et al. (cited in the paper as [4]) and is used to draw the dashed reference lines in the plots. 

### Training 

The file `sidarthe_exp.py` contains the training code, with all hyperparameters set as the ones that gave the results shown in the paper. To launch the training, it is enough to run the command:

`python sidarthe_exp.py`

It will create two directories: **runs** and **regioni**. **runs** contains the files written by tensorboard, which can be observed in real time with the command 

`tensorboard --logdir runs`

The command will launch a tensorboard instance on the default endpoint which is http://localhost:6006.

**regioni** contains a tree of directories, browse it until you get to **regioni/sidarthe/Italy/**. Here you will find one or more directories (based on how many times you launched the script) with unique GUIDs. Each of these directories will contain two files: *settings.json* and *final.json*. In particular, *final.json* will be present only after the training ends. 

* *settings.json* contains the hyper parameters used for that experiment
* *final.json* contains the learned parameters, the epoch on which the best validation risk was obtained, and the risks over each data split (i.e. training, validation, test, whole dataset)

### Evaluation

Evaluation scripts come in two files: *evaluation.py* is a simple python script which can be run with: 

`python evaluation.py`

The script will evaluate the trained model that we provide in the directory trained_models. If you want to evaluate your model, please change line 88 with the experiment id after you copied the directory in trained_models.

The script also comes as a jupyter notebook *evaluation.ipynb*.


## Results

Command: `python sidarthe.py`

Results table:

|   Train   |  Validation  |  Test   |  Total  |
| :-------: | :----------: | :-----: | :-----: |
| 1703.7    |    11463.1   | 36456.8 | 17173.6 |