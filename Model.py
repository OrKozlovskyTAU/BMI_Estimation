
import torch
from torch import nn
from Nets import load_resnet, load_efficient
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from transformers import ViTForImageClassification, CvtForImageClassification
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights



# Default values
ACTIVATION_FUNCTION = nn.GELU()
RESNET_NUM = 'resnet101'
EFF = '2'
DROP_PROB = 0
CLASSIFICATION = False
OUT_FEATURES = 5


class Fully_Connected_Layer(nn.Module):
    def __init__(self, in_features: int = 1000,
                 out_features: int = OUT_FEATURES, 
                 activation_fn: torch.nn = ACTIVATION_FUNCTION,
                 drop_prob: float = DROP_PROB,
                 classification: bool = CLASSIFICATION) -> None:
        super().__init__()
        
        if classification:
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                activation_fn,
                nn.Dropout(drop_prob),
                nn.Linear(512, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, 64),
                activation_fn,
                nn.Linear(64, 32),
                activation_fn,
                nn.Linear(32, 16),
                activation_fn,
                nn.Linear(16, out_features)
                # activation_fn,
                # nn.Linear(8, 5)
                )
        
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                activation_fn,
                nn.Dropout(drop_prob),
                nn.Linear(512, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, 64),
                activation_fn,
                nn.Linear(64, 32),
                activation_fn,
                nn.Linear(32, 16),
                activation_fn,
                nn.Linear(16, 1)
                # activation_fn,
                # nn.Linear(8, 1)
                )

    def forward(self, x):
        x = self.fc(x)
        return x

########################################################################################################
    
# Building the model
class ResNet_Model(nn.Module):
    def __init__(self, type_select: torch.hub = RESNET_NUM,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # Load ResNet as the base model
        self.resnet = load_resnet(model_name=type_select)

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      self.resnet.relu,
                                      self.resnet.maxpool,
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        
        # # drop-out layer
        # self.dropout = nn.Dropout(drop_prob)

        # average pooling layer
        self.avgpool = self.resnet.avgpool
        
        # classifier
        self.classifier = self.resnet.fc
        
        # gradient placeholder
        self.gradient = None

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

        # # Set requires_grad=True for parameters
        # for param in self.parameters():
        #     param.requires_grad = True


    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):


        # extract the features (first part of resnet forward pass)
        x = self.features(x)
        
        # register the hook
        x.requires_grad_(True)
        h = x.register_hook(self.activations_hook)

        # complete the rest of the forward pass
        x = self.avgpool(x)
        x = x.view((x.size(0), -1)) # reshaping the tensor x to be a tensor of dim 2 with [batch, feature map] as the fc layer requires in it's input
        x = self.classifier(x)
        x = self.fc(x)
        
        return x

########################################################################################################

# Building the EfficientNet model
class EfficientNet_Model(nn.Module):
    def __init__(self, type_select: torch.hub = EFF,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # EfficientNet b3 architecture
        self.efficient = load_efficient(model_name=type_select)
        
        # isolate the feature blocks
        self.features = self.efficient.features

        # average pooling layer
        self.avgpool = self.efficient.avgpool
        
        # classifier
        self.classifier = self.efficient.classifier
        
        # gradient placeholder
        self.gradient = None

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):
        
        # extract the features (first part of resnet forward pass)
        x = self.features(x)
        # layer_test = x
        # register the hook
        x.requires_grad_(True)
        h = x.register_hook(self.activations_hook)

        # complete the rest of the forward pass
        x = self.avgpool(x)
        x = x.view((x.size(0), -1)) # reshaping the tensor x to be a tensor of dim 2 with [batch, feature map] as the fc layer requires in it's input
        x = self.classifier(x)
        x = self.fc(x)

        return x

########################################################################################################

# Building the MobileNet model
class MobileNet_Model(nn.Module):
    def __init__(self,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # Mobile net v3 architecture
        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # isolate the feature blocks
        self.features = self.mobilenet.features

        # average pooling layer
        self.avgpool = self.mobilenet.avgpool
        
        # classifier
        self.classifier = self.mobilenet.classifier

        # gradient placeholder
        self.gradient = None

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):
        
        # extract the features (first part of resnet forward pass)
        x = self.features(x)
        
        # register the hook
        x.requires_grad_(True)
        h = x.register_hook(self.activations_hook)

        # complete the rest of the forward pass
        x = self.avgpool(x)
        x = x.view((x.size(0), -1)) # reshaping the tensor x to be a tensor of dim 2 with [batch, feature map] as the fc layer requires in it's input
        x = self.classifier(x)
        x = self.fc(x)

        return x

########################################################################################################

# Building the VitTrans model
class VitTransformer_Model(nn.Module):
    def __init__(self,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # Load ViT
        self.vit_tf = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):
                
        x = self.vit_tf(x).logits

        x = self.fc(x)

        return x

########################################################################################################

# Building the Cvt model
class CVT_Transformer_Model(nn.Module):
    def __init__(self,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # Load CVT transformer
        self.cvt_tf = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):
                
        x = self.cvt_tf(x).logits

        x = self.fc(x)

        return x
    
########################################################################################################

# Building the  model
class DenseNet_Model(nn.Module):
    def __init__(self,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # Load ViT
        self.dense = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):
                
        x = self.dense(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

########################################################################################################

# Building the  model
class RegNet_Model(nn.Module):
    def __init__(self,
                 out_features=OUT_FEATURES,
                 activation_fn=ACTIVATION_FUNCTION,
                 drop_prob=DROP_PROB,
                 classification=CLASSIFICATION) -> None:
        super().__init__()

        # Load ViT
        self.regnet = regnet_y_128gf(weights=RegNet_Y_128GF_Weights.DEFAULT)

        # fully connected layer
        self.fc = Fully_Connected_Layer(out_features=out_features,
                                        activation_fn=activation_fn,
                                        drop_prob=drop_prob,
                                        classification=classification)

    # Forward pass in segments so that a hook can be attached
    def forward(self, x):
                
        x = self.regnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

########################################################################################################

# # instantiating the model
# model = MobileNet_Model()

# # printing the models architecture
# print(model.state_dict)