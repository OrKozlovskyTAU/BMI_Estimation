# importing pretrained resnet networks
import torch
import torch.nn as nn
# import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights


def load_resnet(model_name):
    if model_name == 'resnet18':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)
    
    elif model_name == 'resnet34':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=ResNet34_Weights.DEFAULT)
    
    elif model_name == 'resnet50':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
    
    elif model_name == 'resnet101':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights=ResNet101_Weights.DEFAULT)
    
    elif model_name == 'resnet152':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', weights=ResNet152_Weights.DEFAULT)
    
    else:
        raise ValueError("Unsupported ResNet model name")
    
    
def load_efficient(model_name):
    if model_name == '1':
        return efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    
    if model_name == '2':
        return efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    
    elif model_name == '3':
        return efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    
    elif model_name == '4':
        return efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    
    elif model_name == 's':
        return efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    
    elif model_name == 'm':
        return efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    
    elif model_name == 'l':
        return efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    
    else:
        raise ValueError("Unsupported Efficient model name")