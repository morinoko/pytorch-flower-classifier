import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def load_datasets(data_dir):
    """Loads datasets for training, validation, and testing
       Datasets are returned in as a dictionary
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define image transforms for the training, validation, and testing sets
    shared_transforms = [transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])]

    data_transforms = {
        'training': transforms.Compose(
            [transforms.RandomRotation(30),
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()] + 
             shared_transforms),
        'nontraining': transforms.Compose(
            [transforms.Resize(250),
             transforms.CenterCrop(224)] + 
             shared_transforms)
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, data_transforms['nontraining']),
        'test': datasets.ImageFolder(test_dir, data_transforms['nontraining'])
    }
    
    return image_datasets

def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns a Tensor
    """
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    thumb = thumbnail_image(image, 256)
    cropped_image = square_center_crop(thumb, 224)
    
    # Convert image to numpy array of floats
    np_image = np.array(cropped_image)
    
    # Normalize values by dividing by subracting means [0.485, 0.456, 0.406]
    # and dividing by std. dev. [0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image/255 - mean)/std # divide by 255 or the colors turn out weird
    
    # move color channel to first position in array and convert to tensor
    return torch.Tensor(normalized_image.transpose((2, 0, 1)))

def thumbnail_image(image, new_shortest_side):
    """Returns thumbnail of image resized so that the 
       shortest side is resized to the given size, keeping the aspect ration
    
        Args:
            image = image object
            new_shortest_side = desired size of the shortest size in pixels
    """
    size = np.array(image.size)
    shortest_side = size.min()
    aspect_ratio = new_shortest_side/size.min()
    
    return image.resize(np.array(np.floor(size * aspect_ratio), dtype='int'))

def square_center_crop(image, square_size):
    """Returns image cropped to the square of desired size
    """
    
    """
    Args:
        image = image object
        square_size = size in pixels
    """
    width, height = image.size
    
    # Crop box coords
    left = (width - square_size)//2
    top = (height- square_size)//2
    right = left + square_size
    bottom = top + square_size
    
    return image.crop((left, top, right, bottom))