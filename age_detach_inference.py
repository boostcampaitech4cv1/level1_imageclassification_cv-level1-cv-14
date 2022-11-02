import argparse
import multiprocessing
import os
from importlib import import_module
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import age_TestDataset
from PIL import Image
import albumentations as A

from coral import resnet34

def get_age_class(age):
    return 0 if age + 18 < 30 else 1 if age + 18 < 58 else 2

def load_model(saved_model, num_classes, device, use_se):
    if use_se:
        model_cls = getattr(import_module("model"), 'SENet154')
    else:
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
    model = load_model('./model/6class_ori', 6, device,False).to(device)
    model.eval()
    
    #age_model = resnet34(55, False).cuda()
    #age_model.load_state_dict(torch.load('./model/CORAL2020/best_model.pt', map_location='cuda'))
    age_model = load_model('./model/43_class_senet', 43, device, True).to(device)
    age_model.eval()
    
    age_model2 = load_model('./model/43_class_vit_ori', 43, device, False).to(device)
    age_model2.eval()
    
    img_root = os.path.join('../EDA/eval', 'images')
    info_path = os.path.join('../EDA/eval', 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = age_TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    age_preds = []
    with tqdm(loader) as pbar:
        for idx, images in enumerate(pbar):
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                pred = pred.cpu()
                
                age_pred = age_model(images)
                age_pred += age_model2(images)
                age_num = age_pred.argmax(dim=-1)
                age_num = age_num.cpu()
                
                age_preds.extend(age_num.cpu().numpy())
                pred = pred * 3 + age_num.apply_(lambda x: get_age_class(x))
                preds.extend(pred.cpu().numpy())
            pbar.set_description("processing %s" % idx)

    info['ans'] = preds
    info['age'] = age_preds
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
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '../input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/6_class_vit_32batch'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output/Age_Detach_43_ensenble_under58_crop'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
