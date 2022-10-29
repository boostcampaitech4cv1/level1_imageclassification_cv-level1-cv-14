import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CosineSimilarity as CosSim
from torchvision.transforms import Resize, Normalize, Compose
from torchvision.models import efficientnet_b4, efficientnet_b7

#pip install git+https://github.com/openai/CLIP.git
import clip 
from transformers import CLIPProcessor, CLIPModel
#https://github.com/openai/CLIP

#pip install vit_pytorch
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

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
class CustomClipLinear(nn.Module): # 쓰레기
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.features = ["mask", "face", "nose", "cheek", "mouth", "chin", 'lips', "male", "man", "female", "woman", "under 30", "over 30 and under 60", "over 60"]
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.transform = Compose([
            # Resize((224, 224)), # 필요하면
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.net = nn.Sequential(
            nn.Linear(len(self.features), 196),
            nn.BatchNorm1d(196),
            nn.LeakyReLU(0.05),
            nn.Linear(196, self.num_classes),
        )


    def set_features(self, features):
        self.features = features
        self.linear = nn.Linear(len(self.features), self.num_classes)

    def _get_cosine_score(self, imgs):
        # GPU 메모리 약 1.5 GB 필요 --> 만일 부족하다면 clip.available_models() 명령어를 통해 가지고 오는 모델을 바꿀 수 있습니다
        imgs = self.transform(imgs)
        text = clip.tokenize(self.features).to(self.device)
        with torch.no_grad():
            # 모델에 image와 text 둘 다 input으로 넣고, 각 text와 image와의 유사도를 구합니다. 값이 클수록 유사합니다.
            logits_per_image, _ = self.clip_model(imgs, text) # RGB (ex : (1, 3, 244, 244))
            # 확률값으로 표현하기 위해 softmax 값을 구합니다.
            probs = logits_per_image.softmax(dim=-1)
        
        return probs.float()

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x_ = self._get_cosine_score(x)
        out = self.net(x_)
        return out


class EfficientB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.efficient = efficientnet_b4(pretrained=True)
        for param in self.efficient.parameters():
            param.requires_grad = False
        self.efficient.classifier[1] = nn.Linear(1792, self.num_classes)
        # self.efficient.classifier = nn.Sequential(
        #     nn.Linear(1792, 1792),
        #     nn.LeakyReLU(),
        #     nn.Dropout1d(0.4),
        #     nn.Linear(1792, self.num_classes),
        # )

    def forward(self, x):
        out = self.efficient(x)
        return out


class EfficientB7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.efficient = efficientnet_b7(pretrained=True)
        for param in self.efficient.parameters():
            param.requires_grad = False
        self.efficient.classifier[1] = nn.Linear(2560, self.num_classes)
        # self.efficient.classifier = nn.Sequential(
        #     nn.Linear(2560, 2560),
        #     nn.LeakyReLU(),
        #     nn.Dropout1d(0.4),
        #     nn.Linear(2560, self.num_classes),
        # )

    def forward(self, x):
        out = self.efficient(x)
        return out
    
class MyVit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.vit = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
    def forward(self,x):
        out = self.vit(x)
        return out
    

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
    
class T4073_CLIP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features_mask = ["I can see the mouth", "There is no mask in photo", "I can see the nose", "mask covered nose and mouth"]
        self.features_gender = ["male", "man","boy","grand father" "female", "woman","girl", "grand mother"]
        self.features_age = [ "Person in photo looks like " + str(i) + " years old" for i in range(101)]
        self.num_classes = num_classes
        self.device = "cuda"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.fc = nn.Linear(len(self.features_mask) + len(self.features_gender) + len(self.features_age), self.num_classes)
        
    def _get_clip_embedding(self, imgs):
        with torch.no_grad():
            text_mask = clip.tokenize(self.features_mask).to(self.device)
            text_gender = clip.tokenize(self.features_gender).to(self.device)
            text_age = clip.tokenize(self.features_age).to(self.device)
            input_mask = self.clip_preprocess(text=self.features_mask, images=imgs, return_tensors="pt", padding=True)
            input_gender = self.clip_preprocess(text=self.features_mask, images=imgs, return_tensors="pt", padding=True)
            input_age = self.clip_preprocess(text=self.features_mask, images=imgs, return_tensors="pt", padding=True)
            logits_per_image_mask = self.clip_model(input_mask) # RGB (ex : (1, 3, 244, 244))
            logits_per_image_gender = self.clip_model(input_gender)
            logits_per_image_age = self.clip_model(input_age)
            logits_per_image = torch.cat([logits_per_image_mask, logits_per_image_gender, logits_per_image_age], dim = -1)
        
        return logits_per_image

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x_ = self._get_clip_embedding(x)
        out = self.net(x_)
        return out