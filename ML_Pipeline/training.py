from detectron2.utils.logger import setup_logger

setup_logger()
# detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from ML_Pipeline.admin import output_path
import os
from pathlib import Path


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


class Detectron2Training:
    def __init__(self, data_path):
        # base directory to data
        base_path_data = Path(data_path)
        base_path_image = base_path_data.joinpath('images')

        self.train_path = base_path_image.joinpath('train')
        self.train_path.mkdir(parents=True, exist_ok=True)

        self.annot_path = base_path_data.joinpath('annotations')
        self.annot_path.mkdir(parents=True, exist_ok=True)

    def train(self):
        register_name = 'detection_segmentaion'
        annotation = str(self.annot_path.joinpath('annotations.json'))
        train = str(self.train_path)
        register_coco_instances(register_name,
                                {},
                                annotation,
                                train)

        cfg = get_cfg()

        cfg.MODEL.DEVICE = 'cpu'
        # Load values from a file
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        # Add new configs custom components
        cfg.DATASETS.TRAIN = (register_name,)  # training images; remember to add ",", dtype: tuple
        cfg.DATASETS.TEST = ()  # validation images; remember to add ",", dtype: tuple
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2  # images per batch
        cfg.SOLVER.BASE_LR = 0.0001  # mininum learning rate

        cfg.SOLVER.WARMUP_ITERS = 10
        cfg.SOLVER.MAX_ITER = 100

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        cfg.TEST.EVAL_PERIOD = 10
        cfg.OUTPUT_DIR = output_path
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # default config from coco
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=True)
        # Starts training here
        trainer.train()

