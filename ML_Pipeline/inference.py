from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from ML_Pipeline.admin import output_path
import cv2
import os
import matplotlib.pyplot as plt


class Detectron2Infer:
    def __init__(self):
        self.cfg = get_cfg()

        self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATALOADER.NUM_WORKERS = 4

        self.cfg.SOLVER.IMS_PER_BATCH = 2  # images per batch
        self.cfg.OUTPUT_DIR = output_path

        self.output_infer = os.path.join(output_path, "output_model")
        if not os.path.exists(self.output_infer):
            os.makedirs(self.output_infer, 0o755)

        self.cfg.SOLVER.BASE_LR = 0.02  # mininum learning rate
        self.cfg.SOLVER.WARMUP_ITERS = 1000
        self.cfg.SOLVER.MAX_ITER = 2000
        self.cfg.SOLVER.STEPS = (1000, 1500)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 164
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 1 classes (rot)
        self.cfg.TEST.EVAL_PERIOD = 100

    def infer(self, image_path):
        # Reading the image
        im = cv2.imread(image_path)
        # load the model here
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set the testing threshold for this model
        predictor = DefaultPredictor(self.cfg)
        outputs_inference_image_2 = predictor(im)
        # using visualizer util from detectron to write the output into file

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=0.8)
        out = v.draw_instance_predictions(outputs_inference_image_2["instances"].to("cpu"))
        out.save(os.path.join(self.output_infer, './test_image_1_inference.jpg'))
        return outputs_inference_image_2
