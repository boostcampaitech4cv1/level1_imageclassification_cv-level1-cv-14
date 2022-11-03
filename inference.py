import argparse
import multiprocessing
import os
from importlib import import_module
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TTA, ImageToTensor, TestDataset, MaskBaseDataset, ValidAugmentation
from torchvision.transforms import Resize
import numpy as np


def get_age_class(age):
    return 0 if age < (30-18) else 1 if age < (59-18) else 2


def load_model(saved_model, num_classes, device, model_name):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    if args.ensemble:
        # ViT384
        vit_model = load_model('./model/vit384_mingi', 6, device, 'MyVit384').to(device)
        vit_model.eval()
        vit_resize = Resize([384, 384])
        # SwinV2
        # swin_model = load_model('./model/swinv2-alb-2', num_classes, device, 'SwinV2').to(device)
        # swin_model.eval()
        # swin_resize = Resize([224, 224])
        # Efficient B4
        efficient_model = load_model('./model/effib4-alb-2', num_classes, device, 'EfficientB4').to(device)
        efficient_model.eval()
        efficient_resize = Resize([380, 380])
        # SENet
        senet_model = load_model('./model/senet-alb-2', num_classes, device, 'SENet154').to(device)
        senet_model.eval()
        senet_resize = Resize([224, 224])
        # MixNet
        mixnet_model = load_model('./model/mixnet-alb-2', num_classes, device, 'MixNet').to(device)
        mixnet_model.eval()
        mixnet_resize = Resize([224, 224])
        args.resize = None
    else:
        model = load_model(model_dir, num_classes, device, args.model).to(device)
        model.eval()
        
    if args.age_model:
        # ViT(only age)
        age_vit = load_model('./model/43_class_vit_ori', 43, device, 'MyVit384').to(device)
        age_vit.eval()
        age_senet = load_model('./model/43_class_senet', 43, device, 'SENet154').to(device)
        age_senet.eval()
        # age_facenet = load_model('./model/43_facenet', 43, device, 'InceptionResnet').to(device)
        # age_facenet.eval()
        facenet_resize = Resize([112, 112])

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    print("Calculating inference results..")
    preds = []
    
    with tqdm(loader) as pbar:
        for idx, images in enumerate(pbar):
            with torch.no_grad():
                images = images.to(device)
                # sum pred
                if args.ensemble:
                    pred = vit_model(vit_resize(images))
                    pred = torch.repeat_interleave(pred, 3, dim=1) # classes 6 x 3
                    # pred = swin_model(swin_resize(images))
                    pred += efficient_model(efficient_resize(images))
                    pred += senet_model(senet_resize(images))
                    # pred += mixnet_model(mixnet_resize(images))
                else:
                    pred = model(images)
                    
                pred = pred.argmax(dim=-1)
                pred = pred.cpu()
                
                if args.age_model:
                    # age pred
                    if args.ensemble:
                        age_pred = age_vit(vit_resize(images))
                        age_pred += age_senet(senet_resize(images))
                    else:
                        age_pred = age_vit(images)
                    age_pred = age_pred.argmax(dim=-1)
                    age_pred = age_pred.cpu()
                    age_pred = age_pred.apply_(lambda x: get_age_class(x))
                    # calc classes
                    mask_pred = torch.div(pred, 6, rounding_mode='trunc') % 3
                    gender_pred = torch.div(pred, 3, rounding_mode='trunc') % 2
                    pred = mask_pred * 6 + gender_pred * 3 + age_pred
                
                preds.extend(pred.cpu().numpy())
            pbar.set_description("processing %s" % idx)
    
    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")
    
    
@torch.no_grad()
def validation(data_dir, model_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    if args.ensemble:
        # ViT384
        vit_model = load_model('./model/vit384_donghun', num_classes, device, 'MyVit384').to(device)
        vit_model.eval()
        vit_resize = Resize([384, 384])
        # SwinV2
        vit2_model = load_model('./model/swinv2-alb-2', num_classes, device, 'SwinV2').to(device)
        vit2_model.eval()
        vit2_resize = Resize([224, 224])
        # Efficient B4
        efficient_model = load_model('./model/effib4-alb-2', num_classes, device, 'EfficientB4').to(device)
        efficient_model.eval()
        efficient_resize = Resize([380, 380])
        # SENet
        senet_model = load_model('./model/senet-alb-2', num_classes, device, 'SENet154').to(device)
        senet_model.eval()
        senet_resize = Resize([224, 224])
        # MixNet
        mixnet_model = load_model('./model/mixnet-alb-2', num_classes, device, 'MixNet').to(device)
        mixnet_model.eval()
        mixnet_resize = Resize([224, 224])
        args.resize = None
    else:
        model = load_model(model_dir, num_classes, device, args.model).to(device)
        model.eval()
    
    dataset = MaskBaseDataset(data_dir='/opt/ml/input/data/train/images')
    _, val_set = dataset.split_dataset()
    transform = ValidAugmentation(resize=None) # ImageToTensor
    val_set.dataset.set_transform(transform)
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    with torch.no_grad():
        print("Calculating validation results...")
        mask_acc_items = []
        gender_acc_items = []
        with tqdm(val_loader) as pbar:
            for idx, val_batch in enumerate(pbar):
                images, labels = val_batch
                images = images.to(device)
                labels = labels.to(device)
                
                # sum pred
                if args.ensemble:
                    pred = vit_model(vit_resize(images))
                    # pred = torch.repeat_interleave(pred, 3, dim=1)
                    # pred += vit2_model(vit2_resize(images))
                    pred += efficient_model(efficient_resize(images))
                    pred += senet_model(senet_resize(images))
                    pred += mixnet_model(mixnet_resize(images))
                else:
                    pred = model(images)
                    
                pred = pred.argmax(dim=-1)
                mask_pred = torch.div(pred, 6, rounding_mode='trunc') % 3
                gender_pred = torch.div(pred, 3, rounding_mode='trunc') % 2
                mask_label = torch.div(labels, 6, rounding_mode='trunc') % 3
                gender_label = torch.div(labels, 3, rounding_mode='trunc') % 2
                
                mask_acc_item = (mask_label == mask_pred).sum().item()
                mask_acc_items.append(mask_acc_item)
                gender_acc_item = (gender_label == gender_pred).sum().item()
                gender_acc_items.append(gender_acc_item)
            pbar.set_description("processing %s" % idx)

        mask_acc = np.sum(mask_acc_items) / len(val_set)
        gender_acc = np.sum(gender_acc_items) / len(val_set)
        print(f"Mask val accuracy : {mask_acc:4.2%}")
        print(f"Gender val accuracy : {gender_acc:4.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: [128, 96])')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--ensemble', type=bool, default=False)
    parser.add_argument('--age_model', type=bool, default=False)
    parser.add_argument('--valid', type=bool, default=False)

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.valid:
        validation(data_dir, model_dir, args)
    else:
        inference(data_dir, model_dir, output_dir, args)
