import os
from PIL import Image
import pandas as pd
from rembg import remove

#input_path = "../../EDA/train/images"
#output_path = "../../EDA/train/removed_images"
#df = pd.read_csv('../../EDA/train/train.csv')

input_path = "../../EDA/eval/images"
output_path = "../../EDA/eval/removed_images"
df = pd.read_csv('../../EDA/eval/info.csv')

def get_ext(img_dir, img_id, idx):
    filename = os.listdir(os.path.join(img_dir, img_id))[idx]
    ext = os.path.splitext(filename)[-1].lower()
    return ext

for i in range(len(df)):
    #if i < 2399:
    #    continue
    #img_id = df.iloc[i].path
    #for idx, img_name in enumerate(['incorrect_mask', 'mask1', 'mask2', 'mask3','mask4','mask5', 'normal']):
    #    img_dir = os.path.join(input_path, img_id,img_name)
    #    ext = get_ext(input_path, img_id,idx)
    #    print(img_dir+ext)
    #    img = Image.open(img_dir+ext)
    #    output = remove(img).convert('RGB')
    #    output_dir = os.path.join(output_path, img_id, img_name+'.jpg')
    #    print(output_dir)
    #    output.save(output_dir)
    if i < 2634:
        continue
    img_id = df.iloc[i].ImageID
    img_dir= os.path.join(input_path, img_id)
    print(img_dir)
    img = Image.open(img_dir)
    output = remove(img).convert('RGB')
    output_dir = os.path.join(output_path,img_id)
    print(output_dir)
    output.save(output_dir)
    
