import torch
import json
from input_args_helper import get_prediction_args
from model import Classifier, load_checkpoint
from image_processing_helper import process_image

def main():
    # Get arguments from CLI input
    # image: path to image file (req.)
    # checkpoint: path to model checkpoint file (req.)
    # top_k: number of top K predictions to show (default 3)
    # category_names: file for mapping of categories to real name (optional)
    # gpu: switch GPU on (default False)
    input_args = get_prediction_args()

    model = load_checkpoint(input_args.checkpoint, input_args.gpu)
    model.eval()
    
    image = process_image(input_args.image)
    # Add 'batch' parameter of 1 to image tensor needed by the model
    # Image size becomes [1, 3, 224, 224]
    image = image.unsqueeze(0)
    
    # Make prediction
    probabilities = torch.exp(model(image))
    top_probs, class_indices = [tensor.tolist()[0] for tensor in probabilities.topk(input_args.top_k, dim=1)]
    
    # Map class indices to class labels (from image folders)
    class_dict = model.class_to_idx
    class_list = list(class_dict.keys())
    index_list = list(class_dict.values())

    predicted_classes = [class_list[index_list.index(idx)] for idx in class_indices]
    
    # Use labels from a category mapping JSON files if given
    if input_args.category_names:        
        predicted_classes = convert_cats_to_names(input_args.category_names, predicted_classes)
    
    print_results(top_probs, predicted_classes)


def convert_cats_to_names(cat_to_name_json, categories):
    """Returns list of names converted from a list of categories
    """
    
    """
    Args:
        cat_to_name_json: path to json file containing category to name mappings
        categories: list of categories to convert to names
        
    Returns: list of names
    """
    
    with open(cat_to_name_json, 'r') as f:
            cat_to_name = json.load(f)
            
    return [cat_to_name[key] for key in categories]


def print_results(probabilities, predicted_classes):
    n_predictions = len(probabilities)
    
    if len(probabilities) > 1:
        print(f'The top {n_predictions} predictions are:')

        for i in range(n_predictions):
            print(f'  {i+1}. {predicted_classes[i].title()} with {round(probabilities[i], 3)} probablility')
    else:
        print(f'The top prediction is {predicted_classes[0]} with a probability of {round(probabilities[0], 3)}')
                  
    
if __name__ == '__main__':
    main()
    