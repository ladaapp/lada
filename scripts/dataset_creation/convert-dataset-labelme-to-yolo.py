import glob
import os
import json
import argparse
import shutil

def convert_to_yolo_txt_lines(labelme_json, labelme_label_to_yolo_class_mapping={"nsfw": 0}):
    image_height = labelme_json["imageHeight"]
    image_width = labelme_json["imageWidth"]
    
    point_txt = []
    
    for shape in labelme_json["shapes"]:
        if shape["label"] not in labelme_label_to_yolo_class_mapping:
            continue
        if shape["shape_type"] != "polygon":
            continue

        yolo_class = labelme_label_to_yolo_class_mapping[shape["label"]]
        txt = f"{yolo_class}"
        
        for w, h in  shape["points"]:
            txt = f'{txt} {float(w)/image_width} {float(h)/image_height}'
        
        point_txt.append(txt)
    return point_txt

def main(input_json_dir, output_text_dir, output_images_dir):
    labelme_json_file_paths = glob.glob(os.path.join(input_json_dir, "*.json"))
    for labelme_json_file_path in labelme_json_file_paths:
        with open(labelme_json_file_path) as labelme_json_file:
            labelme_json = json.load(labelme_json_file)
            image_path = labelme_json['imagePath']
            image_filename = os.path.basename(image_path)

            yolo_img_file_path = os.path.join(output_images_dir, image_filename)
            shutil.copyfile(os.path.join(input_json_dir, image_path), yolo_img_file_path)
            
            yolo_txt_lines = convert_to_yolo_txt_lines(labelme_json)
            if len(yolo_txt_lines) == 0:
                continue

            image_filename_without_ext = os.path.splitext(image_filename)[0]
            yolo_txt_file_path = os.path.join(output_text_dir, image_filename_without_ext + '.txt')
            with open(yolo_txt_file_path, 'w') as yolo_txt_file:
                for line in yolo_txt_lines:
                    yolo_txt_file.write(line)
                    yolo_txt_file.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-in', type=str)
    parser.add_argument('--dir-out-images', type=str)
    parser.add_argument('--dir-out-labels', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.dir_in, args.dir_out_labels, args.dir_out_images)

# mkdir -p yolo/datasets/nsfw_detection/{train,val}/{images,labels}
# python yolo/convert-dataset-labelme-to-yolo.py --dir-in datasets/nsfw_detection/val --dir-out-images yolo/datasets/nsfw_detection/val/images --dir-out-labels yolo/datasets/nsfw_detection/val/labels
# python yolo/convert-dataset-labelme-to-yolo.py --dir-in datasets/nsfw_detection/train --dir-out-images yolo/datasets/nsfw_detection/train/images --dir-out-labels yolo/datasets/nsfw_detection/train/labels
