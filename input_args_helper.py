import argparse

def get_training_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', help = 'Directory that contains the training data')
    
    parser.add_argument('--save_dir', default = '.',
                        help = 'Set directory for saving checkpoints (defaults to current directory)')
    
    parser.add_argument('--arch', default = 'vgg', choices=('vgg', 'densenet', 'alexnet'),
                        help = 'Choose architecture for transfer learning (defaults to vgg)')
    
    parser.add_argument('--learning_rate', default = 0.01, type = float,
                        help = 'Set the learning rate (defaults to 0.01)')
    
    parser.add_argument('--hidden_units', default = 512, type = int,
                        help = 'Set the number of units for the hidden layer (defaults to 512)')
    
    parser.add_argument('--epochs', default = 20, type = int,
                        help = 'Set the number of epochs for training (defaults to 20)')
    
    parser.add_argument('--gpu', action = 'store_true', default = False,
                        help = 'Use GPU for training')
    
    return parser.parse_args()

def get_prediction_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', help = 'Path to the image to predict')
    parser.add_argument('checkpoint', help = 'Path to the checkpoint file for model')
    
    parser.add_argument('--top_k', default = 1, type = int,
                        help = 'Number of top K predictions to return (defaults to 1)')
    
    parser.add_argument('--category_names',
                        help = 'Choose a json file for mapping categories to names')
    
    parser.add_argument('--gpu', action = 'store_true', default = False,
                        help = 'Use GPU for prediction')
    
    return parser.parse_args()
