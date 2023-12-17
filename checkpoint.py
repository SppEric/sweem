import torch

def save(path, model, optimizer, epoch_train_losses, epoch_val_losses, settings):
    """
    Saves a PyTorch model, its optimizer, training/validation losses, and settings to the specified path.

    Parameters:
    model (torch.nn.Module): The PyTorch model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer used for training the model.
    path (str): The file path where the model, optimizer, losses, and settings should be saved.
    epoch_train_losses (list): List of training losses for each epoch.
    epoch_val_losses (list): List of validation losses for each epoch.
    settings (dict): The settings dictionary containing model and training configurations.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_losses': epoch_val_losses,
        'settings': settings
    }, path)


def load(path, model_class, optimizer_class):
    """
    Loads a PyTorch model, its optimizer, training/validation losses, and settings from the specified path.

    Parameters:
    path (str): The file path where the model, optimizer, losses, and settings are saved.
    model_class (class): The class of the model to be loaded.
    optimizer_class (class): The class of the optimizer to be used with the model.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model.
    optimizer (torch.optim.Optimizer): The optimizer for the model.
    epoch_train_losses (list): List of training losses for each epoch.
    epoch_val_losses (list): List of validation losses for each epoch.
    settings (dict): The settings dictionary containing model and training configurations.
    """
    checkpoint = torch.load(path)

    settings = checkpoint['settings']
    model = model_class(**settings['model'])
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optimizer_class(model.parameters(), lr=settings['train']['lr'], weight_decay=settings['train']['l2'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_train_losses = checkpoint.get('epoch_train_losses', [])
    epoch_val_losses = checkpoint.get('epoch_val_losses', [])

    return model, optimizer, epoch_train_losses, epoch_val_losses, settings