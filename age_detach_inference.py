import argparse
import multiprocessing
import os
from importlib import import_module
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import age_detach_TestDataset
from PIL import Image
import albumentations as A

from coral import resnet34

def get_age_class(age):
    return 0 if age < 30 else 1 if age < 60 else 2

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

    #num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, 6, device).to(device)
    model.eval()
    
    age_model = resnet34(55, False).cuda()
    age_model.load_state_dict(torch.load('./model/CORAL2020/best_model.pt', map_location='cuda'))
    age_model.eval()
    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = age_detach_TestDataset(img_paths, args.resize)
    age_dataset = age_detach_TestDataset(img_paths, (120,120))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    age_loader = torch.utils.data.DataLoader(
        age_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    age_iter = iter(age_loader)
    print("Calculating inference results..")
    preds = []
    
    with tqdm(loader) as pbar:
        for idx, images in enumerate(loader):
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                age_image = next(age_iter).to(device)
                age_logits, probas = age_model(age_image)
                #print(age_logits)
                predict_levels = probas > 0.5
                print(predict_levels)
                predicted_label = torch.sum(predict_levels, dim=1)
                age_pred = torch.mean(predicted_label,dtype=float) + 16
                #print(predicted_label)
                pred = pred * 3 + get_age_class(age_pred)
                preds.extend(pred.cpu().numpy())
            pbar.set_description("processing %s" % idx)

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[384, 384], help='resize size for image when you trained (default: [384, 384])')
    parser.add_argument('--model', type=str, default='MyVit384', help='model type (default: MyVit384)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '../EDA/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output/Age_Detach_'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
