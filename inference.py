import argparse
import multiprocessing
import os
from importlib import import_module
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TTA, ImageToTensor, TestDataset, MaskBaseDataset, ValidAugmentation
import numpy as np


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
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
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

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
                pred = model(images)
                pred = pred.argmax(dim=-1)
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
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()
    
    dataset = MaskBaseDataset(data_dir='/opt/ml/input/data/train/images')
    _, val_set = dataset.split_dataset()
    transform = ValidAugmentation() # ImageToTensor
    val_set.dataset.set_transform(transform)
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    # if args.ensemble:
    #     center_crop, horizon_flip, center_horizon, image_resize = TTA(args.resize, [320, 256])
    
    with torch.no_grad():
        print("Calculating validation results...")
        model.eval()
        val_acc_items = []
        with tqdm(val_loader) as pbar:
            for idx, val_batch in enumerate(pbar):
                images, labels = val_batch
                images = images.to(device)
                labels = labels.to(device)
                
                # if args.ensemble:
                #     pred = model(image_resize(images)) / 4
                #     pred += model(center_crop(images)) / 4
                #     pred += model(horizon_flip(images)) / 4
                #     pred += model(center_horizon(images)) / 4
                # else:
                #     pred = model(image_resize(images))
                
                pred = model(images)
                preds = torch.argmax(pred, dim=-1)
                
                acc_item = (labels == preds).sum().item()
                val_acc_items.append(acc_item)
            pbar.set_description("processing %s" % idx)

        val_acc = np.sum(val_acc_items) / len(val_set)
        print(f"Best model for val accuracy : {val_acc:4.2%}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: [128, 96])')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--ensemble', type=bool, default=False)
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
        validation(data_dir, model_dir, output_dir, args)
    else:
        inference(data_dir, model_dir, output_dir, args)
