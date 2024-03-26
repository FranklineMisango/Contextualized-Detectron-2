from ML_Pipeline.training import Detectron2Training
from ML_Pipeline.inference import Detectron2Infer
from ML_Pipeline.admin import train_dir
import os

# training detecton2 for object detection and segmentation
detectron2_training_obj = Detectron2Training(train_dir)
detectron2_training_obj.train()

# # inference detectron2
image_path = os.path.join(train_dir, "images", "test", "01-auto.jpg")
detectron2_infer = Detectron2Infer()
outputs_inference_image_2=detectron2_infer.infer(image_path)
print(outputs_inference_image_2['instances'])
