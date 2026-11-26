import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from models.pretrained_vit import load_pretrained_vit, PRETRAINED_WEIGHTS


class LoRA(nn.Module):
    # a weight matrix W of shape (out, in) gets two small matrices:
    # A of shape (rank, in) and B of shape (out, rank), so B @ A has the same shape as W
    def __init__(self, out_features, in_features, rank, alpha):
        super().__init__()
        # A starts with small random values, B with zeros
        # at the start B @ A = 0, so the model behaves exactly like the pretrained one
        self.A = nn.Parameter(torch.randn(rank, in_features) / in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        # alpha / rank controls how strong the LoRA change is
        self.scale = alpha / rank

    def forward(self, W):
        # gets the frozen pretrained weight and returns the changed weight
        return W + (self.B @ self.A) * self.scale


def create_model(num_classes, image_size=224, rank=8, alpha=16, pretrained_weights=PRETRAINED_WEIGHTS):
    # same pretrained ViT L 16 as for linear probing
    model = load_pretrained_vit(image_size, pretrained_weights)

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # add LoRA to the attention layer of all transformer blocks
    for block in model.encoder.layers:
        attention = block.self_attention

        # in_proj_weight holds the query, key and value weights in one matrix
        out_features, in_features = attention.in_proj_weight.shape

        # register_parametrization makes PyTorch run the LoRA module every time the layer
        # uses in_proj_weight, so the layer automatically uses W + B @ A instead of W
        lora = LoRA(out_features, in_features, rank, alpha)
        parametrize.register_parametrization(attention, "in_proj_weight", lora)

    # replace the last layer (1000 ImageNet classes) with a new one for our classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def save_model(path, weights, image_size, rank, alpha, classes):
    # save the model weights (only the LoRA and last layer weights), rank, alpha and some metadata
    checkpoint = {
        "weights": weights, # only the trained weights
        "image_size": image_size,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "rank": rank,
        "alpha": alpha,
        "classes": classes,
    }
    torch.save(checkpoint, path)


def load_model(path, device):
    # build the same model again (same image size, rank, alpha and pretrained weights) and put the saved weights into it
    checkpoint = torch.load(path, map_location="cpu")
    model = create_model(len(checkpoint["classes"]), checkpoint["image_size"], checkpoint["rank"], checkpoint["alpha"], checkpoint["pretrained_weights"])

    # strict=False because the file only has the LoRA and last layer weights, not the whole model
    model.load_state_dict(checkpoint["weights"], strict=False)

    model = model.to(device)
    model.eval()  # we only use it for predictions
    return model, checkpoint
