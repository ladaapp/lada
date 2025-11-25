# SPDX-FileCopyrightText: yolosegment2labelme Authors
# SPDX-License-Identifier: MIT AND AGPL-3.0
# Code vendored from: https://github.com/Abonia1/yolosegment2labelme

import json
import cv2
import os
import argparse

from lada.models.yolo.yolo import Yolo

class PolygonSaver:
    """
    Class to save image information along with polygon shapes and xy coordinates to a JSON file.
    """

    def __init__(self):
        pass

    def save_image_info_with_polygons(self, image_path, xy_coordinates, label_name, output_dir):
        """
        Save image information along with polygon shapes and xy coordinates to a JSON file.
        If the JSON file already exists, append the new mask information to it.

        Args:
            image_path (str): Path to the image file.
            xy_coordinates (list): List of xy coordinates.
            label_name (str): Label name for the polygon shapes.
            output_dir (str): Directory to save the JSON file.
        """
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        json_data = {
            "version": "0.3.3",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "text": ""
        }

        xy_coordinate_as_lists = [xy_coordinate.tolist() for xy_coordinate in xy_coordinates]

        json_data["shapes"].append({
            "label": label_name,
            "text": "",
            "points": xy_coordinate_as_lists,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

        for shape in json_data['shapes']:
            for point in shape['points']:
                shape['points'] = point

        json_file_name = os.path.join(output_dir,
                                      os.path.basename(image_path).replace(os.path.splitext(image_path)[1], ".json"))

        if os.path.exists(json_file_name):
            with open(json_file_name, 'r') as json_file:
                existing_data = json.load(json_file)

            existing_data["shapes"].extend(json_data["shapes"])

            with open(json_file_name, 'w') as json_file:
                json.dump(existing_data, json_file, indent=2)
        else:
            with open(json_file_name, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)

        print(f"Polygon information saved for {image_path} with label {label_name}")

    def generate_json_with_results(self, results, output_dir):
        """
        Generate JSON files with results obtained from YOLO model predictions.

        Args:
            results (list): List of results obtained from YOLO model predictions.
            output_dir (str): Directory to save the JSON files.
        """
        for result in results:
            if result.masks is None:
                continue
            class_labels = result.boxes.cls.cpu().numpy()
            label_names = {k: v for k, v in result.names.items()}

            for mask, class_label in zip(result.masks, class_labels):
                xy_coordinates = mask.xy
                image_path = result.path
                label_name = label_names[int(class_label)]
                self.save_image_info_with_polygons(image_path, xy_coordinates, label_name, output_dir)




def main():
    parser = argparse.ArgumentParser(description='Convert YOLO results to JSON files.')
    parser.add_argument('--model', default='yolo/runs/segment/train_yolov9_c_sgd_5_more_training_data/weights/best.pt', help='Path to YOLO model weights file')
    parser.add_argument('--input', required=True, help='Path to folder containing images')
    parser.add_argument('--output', required=True, help='Path to folder containing images')
    parser.add_argument('--conf', type=float, default=0.2, help='Confidence threshold')
    args = parser.parse_args()

    # Use the input images directory as the output directory for JSON files
    output_dir = args.output

    # Instantiate the PolygonSaver class
    polygon_saver = PolygonSaver()

    # Load the YOLO model
    yolo_model = Yolo(args.model)

    # Get results from YOLO model predictions
    results = yolo_model.predict(args.input, save=False, conf=args.conf)

    # Generate JSON files with results
    polygon_saver.generate_json_with_results(results, output_dir)

if __name__ == "__main__":
    main()