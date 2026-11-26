import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_model(num_classes):
    # load the pretrained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # replace the last layer (1000 ImageNet classes) with a new one for our classes
    # this layer is new, so it's not frozen
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_model(path, fc_weights, image_size, classes):
    # save the model weights (only the last layer) and some metadata
    checkpoint = {
        "weights": fc_weights,
        "image_size": image_size,
        "classes": classes,
    }
    torch.save(checkpoint, path)


def load_model(path, device):
    # build the same model again and put the saved last layer into it
    checkpoint = torch.load(path, map_location="cpu")
    model = create_model(len(checkpoint["classes"]))
    model.fc.load_state_dict(checkpoint["weights"]) # load the last layer weights

    model = model.to(device)
    model.eval()  # we only use it for predictions
    return model, checkpoint
