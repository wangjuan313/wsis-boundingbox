from torchvision_wj.models.segwithbox.residualunet import ResidualUNet
from torchvision_wj.models.segwithbox.enet import ENet

__all__ = ["unet_residual", "enet"]

def unet_residual(input_dim, num_classes, softmax, channels_in=32):
    model = ResidualUNet(input_dim, num_classes, softmax, channels_in)
    return model

def enet(input_dim, num_classes, softmax, channels_in=16):
    model = ENet(input_dim, num_classes, softmax, channels_in)
    return model

