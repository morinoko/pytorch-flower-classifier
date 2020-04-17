import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
from input_args_helper import get_training_args
from model import Classifier, save_checkpoint
from image_processing_helper import load_datasets
from model_input_data import model_input_dict

def main():
    # Get arguments from CLI input
    # data_dir: directory where datasets are located (req.)
    # save_dir: directory to save model's checkpoint file
    # arch: model architecture to use (default vgg11)
    # learning_rate: learning rate for training (default 0.01)
    # hidden_units: units to include in hidden layer (default 512)
    # epochs: number of training epochs (default 20)
    # gpu: switch GPU on (default False)
    input_args = get_training_args()
    
    # Prepare data
    image_datasets = load_datasets(input_args.data_dir)
    training_loader = torch.utils.data.DataLoader(image_datasets['training'], batch_size=64)
    validation_loader = torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64)
    
    # Prepare model and classifier
    model = prepare_model(input_args.arch) 
    
    hyperparams = {
        'inputs': model_input_dict[input_args.arch],
        'hidden': input_args.hidden_units,
        'outputs': len(image_datasets['training'].class_to_idx),
        'class_to_idx': image_datasets['training'].class_to_idx
    }

    classifier = Classifier(hyperparams['inputs'], hyperparams['hidden'], hyperparams['outputs'])

    # Set the model's classifier to be the new, untrained classifier
    model.classifier = classifier

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=input_args.learning_rate)
    
    # Set device
    device = torch.device("cuda:0" if input_args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train model and save checkpoint
    print('Now training...\n')
    
    model.classifier.learn(device, model, training_loader, validation_loader, criterion, optimizer, epochs=input_args.epochs)
    
    save_checkpoint(input_args.save_dir, model, hyperparams)
    
    
def prepare_model(model_name):
    """Returns the proper model based on name with parameters frozen
    """
    
    if model_name == 'vgg':
        model = models.vgg11(pretrained=True)
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        
    # Make sure parameters are frozen
    for param in model.parameters():
        param.requires_grad = False
        
    return model
    
if __name__ == '__main__':
    main()
