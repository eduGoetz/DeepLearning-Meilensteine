from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models.vision_transformer import interpolate_embeddings

PATCH_SIZE = 16  # ViT L 16 cuts every image into 16x16 pixel patches

PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
# PRETRAINED_WEIGHTS = "IMAGENET1K_SWAG_LINEAR_V1"


def load_pretrained_vit(image_size=224, pretrained_weights=PRETRAINED_WEIGHTS):
    # size has to be a multiple of 16
    if image_size % PATCH_SIZE != 0:
        raise ValueError(f"image_size has to be a multiple of {PATCH_SIZE}, got {image_size}")

    # build an empty ViT L 16
    model = vit_l_16(image_size=image_size)

    # load the pretrained weights
    weights = ViT_L_16_Weights[pretrained_weights].get_state_dict(progress=True)

    # interpolate the embeddings of the model, so that it can accept smaller image sizes
    weights = interpolate_embeddings(image_size, PATCH_SIZE, weights)

    # update the empty model with the pretrained weights
    model.load_state_dict(weights)
    return model
