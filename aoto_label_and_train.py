import os

import shutil
from autodistill.detection import CaptionOntology
from ultralytics import YOLO

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # in China , you should


def auto_label():
    # the ontology dictionary has the format {caption: class}
    caption_game = {"person": "person", "head": "head"}
    print(caption_game)

    ontology = CaptionOntology(caption_game)

    original_path = "images"  # original  image path
    datasets_path = "dataset"  # the program generate when auto labeled

    # delete folder if it already exists
    if os.path.exists(datasets_path):
        shutil.rmtree(datasets_path)

    from autodistill_grounded_sam import GroundedSAM

    base_model = GroundedSAM(ontology=ontology)

    dataset = base_model.label(
        input_folder=original_path,
        extension=".jpg",
        output_folder=datasets_path)


def yolo_train():
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training) # for game , YOLOv8n is enough

    results = model.train(data='dataset/data.yaml', epochs=10, imgsz=480, device=0, batch=-1)  # use gpu train  cuda:0


if __name__ == '__main__':
    try:
        auto_label()
        yolo_train()
    except Exception as e:
        print('errors:', e)
        pass
