import argparse
import os
from datetime import datetime

from lada.models.yolo.yolo import Yolo
from lada.utils import ultralytics_utils

ultralytics_utils.set_default_settings()

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a model on a validation dataset')
    parser.add_argument('--model-path')
    parser.add_argument('--dataset-config-path', help="Path to YOLO dataset config file")
    parser.add_argument('--conf', type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument('--imgsz', type=int, default=640, help="Target Image/Frame resolution. Image/Frame will be scaled up/down accordingly. Needs to be a multiple of 32 to match YOLO stride size")
    parser.add_argument('--iou', type=float, default=0.7, help="IoU (Intersection over union) used for NMS (Non-Maximum-Suppression) and to calculate recall metric")
    parser.add_argument('--plot',  default=True, action=argparse.BooleanOptionalAction, help="Plot results (precision, recall curves, confusion matrix etc.)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = Yolo(args.model_path)
    run_name = f"run_{datetime.now().strftime("%Y%m%d_%H%M%S")}"

    results = model.val(plots=True, data=args.dataset_config_path, conf=args.conf, iou=args.iou, imgsz=args.imgsz, name=run_name)

    print(f"Results for {args.model_path}:")
    print(results.to_df())
    print("Confusion Matrix:")
    print(results.confusion_matrix.to_df())
    if args.plot:
        runs_dir = ultralytics_utils.get_settings()["runs_dir"]
        plot_dir = os.path.join(runs_dir, model.task, run_name)
        print("Plotted validation results to", plot_dir)