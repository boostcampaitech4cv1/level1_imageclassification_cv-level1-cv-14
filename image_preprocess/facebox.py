from facenet_pytorch import MTCNN, fixed_image_standardization, training
import os
from PIL import Image
import pandas as pd
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as tt

#input_path = "../../EDA/train/images"
#output_path = "../../EDA/train/crop_images"
#df = pd.read_csv('../../EDA/train/train.csv')

input_path = "../../EDA/eval/images"
output_path = "../../EDA/eval/crop_images"
df = pd.read_csv('../../EDA/eval/info.csv')


def get_ext(img_dir, img_id, idx):
    filename = os.listdir(os.path.join(img_dir, img_id))[idx]
    ext = os.path.splitext(filename)[-1].lower()
    return ext

mtcnn = MTCNN(image_size=384, margin=180)
for i in range(len(df)):
    #img_id = df.iloc[i].path
    #for idx, img_name in enumerate(['incorrect_mask', 'mask1', 'mask2', 'mask3','mask4','mask5', 'normal']):
    #    img_dir = os.path.join(input_path, img_id,img_name)
    #    ext = get_ext(input_path, img_id,idx)
    #    print(img_dir+ext)
    #    img = Image.open(img_dir+ext)
    #    output_dir = os.path.join(output_path, img_id, img_name+'.jpg')
    #    try: 
    #        face = mtcnn(img, save_path=output_dir)
    #        output = to_pil_image(face)
    #    except TypeError:
    #        output = tt.CenterCrop(384)(img)
    #        try:
    #            output.save(output_dir)
    #        except FileNotFoundError:
    #            os.mkdir(os.path.join(output_path, img_id))
    #            output.save(output_dir)
    #        
    #    print(output_dir)

    img_id = df.iloc[i].ImageID
    img_dir= os.path.join(input_path, img_id)
    print(img_dir)
    img = Image.open(img_dir)
    output_dir = os.path.join(output_path,img_id)
    try:
        face = mtcnn(img, save_path=output_dir)
        output = to_pil_image(face)
    except TypeError:
        output = tt.CenterCrop(384)(img)
        try:
            output.save(output_dir)
        except FileNotFoundError:
            os.mkdir(os.path.join(output_path, img_id))
            output.save(output_dir)
    print(output_dir)
