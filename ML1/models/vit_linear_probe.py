import torch
import torch.nn as nn

from models.pretrained_vit import load_pretrained_vit, PRETRAINED_WEIGHTS


def create_model(num_classes, image_size=224, pretrained_weights=PRETRAINED_WEIGHTS):
    # load model
    model = load_pretrained_vit(image_size, pretrained_weights)

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # replace the last layer (1000 ImageNet classes) with a new one for our classes
    # this layer is new, so it's not frozen
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def save_model(path, head_weights, image_size, classes):
    checkpoint = {
        "weights": head_weights, # only the trained weights
        "image_size": image_size,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "classes": classes,
    }
    torch.save(checkpoint, path)


def load_model(path, device):
    # build the same model again (same image size and pretrained weights)
    checkpoint = torch.load(path, map_location="cpu")
    model = create_model(len(checkpoint["classes"]), checkpoint["image_size"], checkpoint["pretrained_weights"])
    model.heads.head.load_state_dict(checkpoint["weights"]) # load the last layer weights

    model = model.to(device)
    model.eval()  # we only use it for predictions
    return model, checkpoint
