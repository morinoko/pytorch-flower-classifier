# Image Classifier Project

This project demonstrates using PyTorch to develop an image classifier for 102 flower species. It incorporates transfer learning through networks provided by the [torchvision model subpackage](https://pytorch.org/docs/stable/torchvision/models.html). 

You can see full details on how the model was trained and how it makes predictions in this [Jupyter notebook](https://github.com/morinoko/pytorch-flower-classifier/blob/master/Image%20Classifier%20Project.ipynb). Images come from [this dataset](https://github.com/morinoko/pytorch-flower-classifier/blob/master/Image%20Classifier%20Project.ipynb) (you can also download the dataset separated into categories [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) thanks to Udacity). This project is based on Udacity's AI Programming with Python Nanodegree.

## Requirements

- Python 3
- PyTorch
- TorchVision
- NumPy

## Usage

### Training a new network

You can train your own network by running the `train.py` script on the command line.

**Basic usage**: 

`python train.py data_directory`

Prints out training loss, validation loss, and validation accuracy as the network trains.

**Options**:

- Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
- Choose architecture: `python train.py data_dir --arch vgg`
  - Available network options: `vgg`, `densenet`, `alexnet`
- Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
- Use GPU for training: `python train.py data_dir --gpu`

Defaults:
 - Save directory: current directory
 - Network: `vgg`
 - Learning rate: `0.01`
 - Hidden units: `512`
 - Epochs: `20`
 - GPU: Set to off by default

 Example: `python train.py flowers --save_dir checkpoints --arch densenet --epochs 30 --gpu`

### Using a checkpoint to make predictions

Given an image and a checkpoint file, you can make predictions with a previously trained network by running the `predict.py` script on the command line.

For the flower identifier, I have trained a network using VGG-11 and saved a checkpoint available in the `vgg11-checkpoint.pth` file.

**Basic usage**: 

`python predict.py /path/to/image checkpoint`

Predicts the name of the flower and gives the probability of that species. Path to image and checkpoint file are required

**Options**:

- Return top `K` most likely classes: `python predict.py input checkpoint --top_k 3`
- Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
- Use GPU for inference: `python predict.py input checkpoint --gpu`

Defaults:
 - Category names: none
 - Top K: `1`
 - GPU: Set to off by default

 Example: `python predict.py flowers/test/52/image_04160.jpg vgg11-checkpoint.pth --top_k 5 --category_names cat_to_name.json`

You could try it with the checkpoint provided in this repository and one of the images from the dataset (or your own)!