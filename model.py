import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CosineSimilarity as CosSim
from torchvision.transforms import Resize, Normalize, Compose
from torchvision.models import efficientnet_b7

from torch.utils.checkpoint import checkpoint

#pip install git+https://github.com/openai/CLIP.git
#import clip 
#from transformers import CLIPProcessor, CLIPModel
#https://github.com/openai/CLIP

#pip install vit_pytorch
#from vit_pytorch import ViT
#from vit_pytorch.extractor import Extractor

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from timm import create_model

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
#class CustomClipLinear(nn.Module): # 쓰레기
#    def __init__(self, num_classes):
#        super().__init__()
#
#        """
#        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
#        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
#        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
#        """
#        self.features = ["mask", "face", "nose", "cheek", "mouth", "chin", 'lips', "male", "man", "female", "woman", "under 30", "over 30 and under 60", "over 60"]
#        self.num_classes = num_classes
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
#        self.transform = Compose([
#            # Resize((224, 224)), # 필요하면
#            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#        ])
#        self.net = nn.Sequential(
#            nn.Linear(len(self.features), 196),
#            nn.BatchNorm1d(196),
#            nn.LeakyReLU(0.05),
#            nn.Linear(196, self.num_classes),
#        )
#
#
#    def set_features(self, features):
#        self.features = features
#        self.linear = nn.Linear(len(self.features), self.num_classes)
#
#    def _get_cosine_score(self, imgs):
#        # GPU 메모리 약 1.5 GB 필요 --> 만일 부족하다면 clip.available_models() 명령어를 통해 가지고 오는 모델을 바꿀 수 있습니다
#        imgs = self.transform(imgs)
#        text = clip.tokenize(self.features).to(self.device)
#        with torch.no_grad():
#            # 모델에 image와 text 둘 다 input으로 넣고, 각 text와 image와의 유사도를 구합니다. 값이 클수록 유사합니다.
#            logits_per_image, _ = self.clip_model(imgs, text) # RGB (ex : (1, 3, 244, 244))
#            # 확률값으로 표현하기 위해 softmax 값을 구합니다.
#            probs = logits_per_image.softmax(dim=-1)
#        
#        return probs.float()
#
#    def forward(self, x):
#        """
#        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#        2. 결과로 나온 output 을 return 해주세요
#        """
#        x_ = self._get_cosine_score(x)
#        out = self.net(x_)
#        return out


class MixNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "mixnet_l"
        self.model = create_model(model_name, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1536, self.num_classes, bias=True)

    def forward(self,x):
        out = self.model(x)
        return out
    
    
class SENet154(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "gluon_senet154"
        self.model = create_model(model_name, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(2048, self.num_classes, bias=True)

    def forward(self,x):
        out = self.model(x)
        return out
    
    
class EfficientB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "efficientnet_b4"
        self.model = create_model(model_name, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1792, self.num_classes, bias=True)

    def forward(self,x):
        out = self.model(x)
        return out


class EfficientB7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.efficient = efficientnet_b7(pretrained=True)
        for param in self.efficient.parameters():
            param.requires_grad = False
        self.efficient.classifier[1] = nn.Linear(2560, self.num_classes)

    def forward(self, x):
        out = self.efficient(x)
        return out
    
#class MyVit(nn.Module):
#    def __init__(self, num_classes):
#        super().__init__()
#        self.num_classes = num_classes
#        self.vit = ViT(
#            image_size = 256,
#            patch_size = 32,
#            num_classes = 1000,
#            dim = 1024,
#            depth = 6,
#            heads = 16,
#            mlp_dim = 2048,
#            dropout = 0.1,
#            emb_dropout = 0.1
#        )
#        
#    def forward(self,x):
#        out = self.vit(x)
#        return out
    

class MyVit2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "vit_base_patch16_224"
        self.vit = create_model(model_name, pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.input_f = self.vit.head.in_features
        self.vit.head = nn.Linear(self.input_f, self.num_classes, bias=True)

    def forward(self,x):
        out = self.vit(x)
        return out
        
        
class MaxVit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "maxvit_rmlp_tiny_rw_256"
        self.maxvit = create_model(model_name, pretrained=True)
        for param in self.maxvit.parameters():
            param.requires_grad = False
        self.maxvit.head.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

    def forward(self,x):
        out = self.maxvit(x)
        return out
    
    
class SwinV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "vit_relpos_base_patch16_clsgap_224"
        self.vit = create_model(model_name, pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.input_f = self.vit.head.in_features
        self.vit.head = nn.Linear(self.input_f, self.num_classes, bias=True)

    def forward(self,x):
        out = self.vit(x)
        return out
    
    
class MyVitSAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "vit_base_patch32_224_sam"
        self.vit = create_model(model_name, pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.input_f = self.vit.head.in_features
        self.vit.head = nn.Linear(self.input_f, self.num_classes, bias=True)

    def forward(self,x):
        out = self.vit(x)
        return out


class MyVit384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "vit_base_patch16_384"
        self.vit = create_model(model_name, pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.input_f = self.vit.head.in_features
        self.vit.head = nn.Linear(self.input_f, self.num_classes, bias=True)

    def forward(self,x):
        out = self.vit(x)
        return out
        
        
class MyVit32_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "vit_base_patch32_384"
        self.vit = create_model(model_name, pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.input_f = self.vit.head.in_features
        self.vit.head = nn.Linear(self.input_f, self.num_classes, bias=True)

    def forward(self,x):
        out = self.vit(x)
        return out
        
        
class EfficientNetV2L(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.efficientnet_v2_l = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_l', pretrained=True, nclass=self.num_classes)
        for param in self.efficientnet_v2_l.parameters():
            param.requires_grad = False
        self.efficientnet_v2_l.head.classifier = nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)

    def forward(self,x):
        out = self.efficientnet_v2_l(x)
        return out
    
#class T4073_CLIP(nn.Module):
#    def __init__(self, num_classes):
#        super().__init__()
#        self.features_mask = ["I can see the mouth", "There is no mask in photo", "I can see the nose", "mask covered nose and mouth"]
#        self.features_gender = ["male", "man","boy","grand father" "female", "woman","girl", "grand mother"]
#        self.features_age = [ "Person in photo looks like " + str(i) + " years old" for i in range(101)]
#        self.num_classes = num_classes
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#        self.clip_model, _  = clip.load("ViT-B/32", device=self.device)
#        self.fc = nn.Linear(len(self.features_mask) + len(self.features_gender) + len(self.features_age), self.num_classes)
#        
#    def _get_clip_embedding(self, imgs):
#        with torch.no_grad():
#            text_mask = clip.tokenize(self.features_mask).to(self.device)
#            text_gender = clip.tokenize(self.features_gender).to(self.device)
#            text_age = clip.tokenize(self.features_age).to(self.device)
#
#            logits_per_image_mask, _= self.clip_model(imgs, text_mask) # RGB (ex : (1, 3, 244, 244))
#            logits_per_image_gender, _ = self.clip_model(imgs, text_gender)
#            logits_per_image_age, _ = self.clip_model(imgs, text_age)
#        
#        return logits_per_image_mask.float(), logits_per_image_gender.float(), logits_per_image_age.float()
#
#    def forward(self, x):
#        """
#        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#        2. 결과로 나온 output 을 return 해주세요
#        """
#        emb_mask, emb_gender, emb_age = self._get_clip_embedding(x)
#        x_ = torch.cat([emb_mask, emb_gender, emb_age], dim = -1)
#        out = self.fc(x_)
#        return out
    
#class MyVit_huge_14_224(nn.Module):
#    def __init__(self, num_classes):
#        super().__init__()
#        self.num_classes = num_classes
#        model_name = "vit_huge_patch14_224_in21k"
#        self.vit = create_model(model_name, pretrained=True)
#        for param in self.vit.parameters():
#            param.requires_grad = False
#        self.input_f = self.vit.head.in_features
#        self.vit.head = nn.Linear(self.input_f, self.num_classes, bias=True)
#
#    def forward(self,x):
#        out = self.vit(x)
#        return out

#seo
using_ckpt = False
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out        

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        #print(self.fc_scale)
        #print("이게??",512 * block.expansion * self.fc_scale)
        #print("이건?",block.expansion)
        
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)#294912
        
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        #print(x.shape)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x

class IResNet_SEO(nn.Module):
    def __init__(self, num_classes):
        super(IResNet_SEO,self).__init__()
        self.num_classes = num_classes
        self.IResNetSEO = IResNet(IBasicBlock, [3, 13, 30, 3])
        self.IResNetSEO.load_state_dict(torch.load('C:\\Users\\ths38\\OneDrive\바탕 화면\\ms1mv3_arcface_r100_fp16\\backbone.pth'))
        for param in self.IResNetSEO.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(in_features=512, out_features=43, bias=True)

    def forward(self,x):
        out = self.IResNetSEO(x)
        out = self.fc1(out)
        return out
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class InceptionResnet(nn.Module):
    def __init__(self, num_classes):
        super(InceptionResnet, self).__init__()
        self.resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=num_classes
        )
    def forward(self,x):
        out = self.resnet(x)
        return out
