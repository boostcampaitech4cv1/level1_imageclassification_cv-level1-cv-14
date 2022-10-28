import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.nn import CosineSimilarity as CosSim
from torchvision.transforms import Resize, Normalize, Compose

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
from coca_pytorch.coca_pytorch import CoCa

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

class MyVit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vit = ViT(
            image_size = 32*32,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )
        self.vit = Extractor(vit, return_embeddings_only = True, detach = False)
        
        self.Linear_1 = nn.Linear(2048, 1024)
        self.Bn_1     = nn.BatchNorm1d(1024)
        self.Relu_1   = nn.LeakyReLU(0.05)
        self.Linear_2 = nn.Linear(1024, 512)
        self.Bn_2     = nn.BatchNorm1d(512)
        self.Relu_2   = nn.LeakyReLU(0.05)
        self.Linear_3 = nn.Linear(512, 18)
        self.Softmax  = nn.Softmax(dim = -1)
        
    def forward(self,x):
        x_ = self.vit(x)
        x_ = torch.flatten(x_, start_dim = 1)
        x_ = self.Linear_1(x_)
        x_ = self.Bn_1(x_)
        x_ = self.Relu_1(x_)
        x_ = self.Linear_2(x_)
        x_ = self.Bn_2(x_)
        x_ = self.Relu_2(x_)
        x_ = self.Linear_3(x_)
        out = self.Softmax(x_)
        return out

# Custom Model Template
class CustomClipLinear(nn.Module): # 쓰레기
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.feature_mask = ["Temple", "Eye", "Ear", "Mouth", "Chin", "Cheek", "Eyebrow", "Forehead", "Hair"]
        self.feature_gender = ["Male", "Man", "Female", "Woman"]
        self.feature_age = ["under 30", "over 30 and under 60", "over 60", "Youth", "Middle-age", "Old-age"]
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.transform = Compose([
            # Resize((224, 224)), # 필요하면
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.net_mask = nn.Sequential(
            nn.Linear(len(self.feature_mask), 81),
            nn.BatchNorm1d(81),
            nn.LeakyReLU(0.05)
        )

        self.net_gender = nn.Sequential(
            nn.Linear(len(self.feature_gender), 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.05)
        )
        
        self.net_age = nn.Sequential(
            nn.Linear(len(self.feature_age), 36),
            nn.BatchNorm1d(36),
            nn.LeakyReLU(0.05)
        )

        self.net_total = nn.Sequential(
            nn.Linear(133, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.05),
            nn.Linear(200, 18),
            nn.Softmax(dim=-1)
        )
    #def set_features(self, features):
    #    self.features = features
    #    self.linear = nn.Linear(len(self.features), self.num_classes)

    def _CLIP(self, imgs, features):
        # GPU 메모리 약 1.5 GB 필요 --> 만일 부족하다면 clip.available_models() 명령어를 통해 가지고 오는 모델을 바꿀 수 있습니다
        imgs = self.transform(imgs)
        text = clip.tokenize(features).to(self.device)
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
        x_mask = self._CLIP(x, self.feature_mask)
        x_mask = self.net_mask(x_mask)
        x_gender = self._CLIP(x, self.feature_gender)
        x_gender = self.net_gender(x_gender)
        x_age = self._CLIP(x, self.feature_age)
        x_age = self.net_age(x_age)
        x_ = torch.concat([x_mask, x_gender, x_age], dim = -1)
        out = self.net_total(x_)
        return out
