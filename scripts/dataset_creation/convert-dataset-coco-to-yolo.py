import contextlib
import json
import cv2
from collections import defaultdict
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x

def convert_coco_json(json_file, yolo_labels_dir, use_segments=False, cls91to80=False):
    json_file = Path(json_file)
    save_dir = Path(yolo_labels_dir)
    save_dir.mkdir(exist_ok=True)

    coco80 = coco91_to_coco80_class()

    #fn = Path(save_dir) / json_file.stem.replace('instances_', '')  # folder name
    #fn.mkdir()
    fn = save_dir
    with open(json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
        img = images['%g' % img_id]
        h, w, f = img['height'], img['width'], img['file_name']

        bboxes = []
        segments = []
        for ann in anns:
            if ann['iscrowd']:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            # Segments
            if use_segments:
                if len(ann['segmentation']) > 1:
                    s = merge_multi_segment(ann['segmentation'])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [cls] + s
                if s not in segments:
                    segments.append(s)

        # Write
        with open((fn / f).with_suffix('.txt'), 'a') as file:
            for i in range(len(bboxes)):
                line = *(segments[i] if use_segments else bboxes[i]),  # cls, box or segments
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-file', type=str)
    parser.add_argument('--yolo-labels-dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
        args = parse_args()
        convert_coco_json(args.coco_file, args.yolo_labels_dir,
                          use_segments=True,
                          cls91to80=True)

# python yolo/convert-dataset-coco-to-yolo.py --coco-file datasets/nsfw_detection/train/coco.json --yolo-labels-dir datasets/nsfw_detection/train/img_mask_yolo
# python yolo/convert-dataset-coco-to-yolo.py --coco-file datasets/nsfw_detection/val/coco.json --yolo-labels-dir datasets/nsfw_detection/val/img_mask_yolo
