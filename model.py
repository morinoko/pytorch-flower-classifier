import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

class Classifier(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        
        self.fc1 = nn.Linear(inputs, hidden)
        self.output = nn.Linear(hidden, outputs)
        
        # Add dropout rate with 50% probability
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # make sure input is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        
        # No dropout for output layer, return log softmax of output
        return F.log_softmax(self.output(x), dim=1)
    
    def learn(self, device, model, training_loader, validation_loader, criterion, optimizer, epochs=30, print_every=100):
        running_loss = 0
        steps = 0
        
        for epoch in range(epochs):
            for images, labels in training_loader:
                steps += 1

                images, labels = images.to(device), labels.to(device)

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                log_prob = model.forward(images)
                loss = criterion(log_prob, labels)
                # Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss
                
                if steps % print_every == 0:
                    # Switch to evaluation mode
                    model.eval()
                    
                    validation_loss, accuracy = model.classifier.validation(device, model, validation_loader, criterion)
                    
                    print(f'Epoch: {epoch+1}/{epochs}..',
                          'Training loss: {:.3f}.. '.format(running_loss/print_every),
                          'Validation loss: {:.3f}..'.format(validation_loss),
                          'Accuracy: {:.3f}%'.format(accuracy*100))
                    
                    # Switch back to training mode
                    model.train()
                    # Reset running loss
                    running_loss = 0
                    
    def validation(self, device, model, validation_loader, criterion):
        validation_loss = 0
        accuracy = 0
        
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                # Calculate log probabilities and loss
                log_prob = model.forward(images)
                loss = criterion(log_prob, labels)
                validation_loss += loss

                # Calculate accuracy
                prob = torch.exp(log_prob)
                top_p, top_class = prob.topk(1, dim=1)
                equalities = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equalities.type(torch.FloatTensor)).item()
        
        validation_loss = validation_loss/len(validation_loader)
        accuracy = accuracy/len(validation_loader)
        
        return validation_loss, accuracy
    
def save_checkpoint(directory, model, hyperparams):
    """Saves a checkpoint for a given model and prints location
    """
    
    """
    Args:
        directory: directory to save checkpoint in
        model: trained model to save the checkpoint for
        hyperparams: dictionary of hyperparameters that includes
                     inputs, hidden layer units, outputs, and class_to_idx
    """
    
    print('\nSaving checkpoint...')
    
    checkpoint = {
        'input_size': hyperparams['inputs'],
        'output_size': hyperparams['outputs'],
        'hidden_input': hyperparams['hidden'],
        'classifier_state_dict': model.classifier.state_dict(),
        'class_to_idx': hyperparams['class_to_idx']
    }
    
    save_dir = f'{directory}/checkpoint.pth'
    torch.save(checkpoint, save_dir)
    
    print(f'Checkpoint saved to {save_dir}')

    
def load_checkpoint(filepath, gpu):
    """Returns a model loaded from a checkpoint file.
       Will print a notice if GPU is requested but not available.
    """
    
    """
    Args:
        filepath: path to the checkpoint file
        gpu: (boolean) True to use GPU, False to use CPU
        
    Returns: list of names
    """
    checkpoint_map_location = 'cuda:0' if (gpu and torch.cuda.is_available()) else 'cpu'
    
    if gpu and not torch.cuda.is_available():
        print("Note: Using CPU because cuda GPU is not available...\n")
        
    checkpoint = torch.load(filepath, map_location = checkpoint_map_location)

    model = models.vgg11(pretrained=True)

    # Make sure parameters are frozen
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = Classifier(checkpoint['input_size'], checkpoint['hidden_input'], checkpoint['output_size'])
    model.classifier = classifier
    
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model