import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.nn import CosineSimilarity as CosSim
from torchvision.transforms import Resize, Normalize, Compose


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
        self.linear = nn.Linear(len(self.features), self.num_classes)
        self.softmax = nn.Softmax()


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
        x_ = self.linear(x_)
        out = self.softmax(x_)
        return out
