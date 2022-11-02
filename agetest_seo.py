from deepface import DeepFace
import numpy as np
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "SFace",
]
for i in range(len(models)):
    embedding = DeepFace.represent(img_path = "C:\\Users\\ths38\\ai_space\\mask_com\\train\\images\\000526_female_Asian_59\\incorrect_mask.jpg",
                               model_name = models[i])
    aa = np.array(embedding)
    print(f"=========model: {models[i]}=============")
    print(aa.shape)

# embedding = DeepFace.represent(img_path = "C:\\Users\\ths38\\ai_space\\mask_com\\train\\images\\000526_female_Asian_59\\incorrect_mask.jpg",
#                                model_name = "SFace")
# aa = np.array(embedding)
# print(aa.shape)