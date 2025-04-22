import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Path to your input video
input_video = "../data/kevins-pinball.mp4"

def vid_inf(vid_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video frame dimensions and FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Detectron2 config setup
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectron2 inference
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        # Visualize results
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(instances)
        frame_out = out.get_image()[:, :, ::-1]

        # Show result
        cv2.imshow("Detectron2 Output", frame_out)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the function
vid_inf(input_video)
