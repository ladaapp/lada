# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from os import path as osp
from pathlib import Path

import cv2
import numpy as np

from lada.models.centerface.centerface import CenterFace
import lada.models.bpjdet.inference as bpjdet
from lada.utils import visualization_utils, image_utils, transforms as lada_transforms, Detections, DETECTION_CLASSES, \
    Image
from lada.utils.box_utils import box_overlap
from lada.datasetcreation.detectors.face_detector import FaceDetector
from lada.datasetcreation.detectors.head_detector import HeadDetector
from lada.datasetcreation.detectors.nsfw_frame_detector import NsfwImageDetector
from lada.utils.threading_utils import clean_up_completed_futures
from lada.utils.ultralytics_utils import convert_segment_masks_to_yolo_labels

from torchvision.transforms import transforms as torchvision_transforms

from lada.utils.image_utils import UnsharpMaskingSharpener
from lada.utils.jpeg_utils import DiffJPEG
from lada.models.yolo.yolo import Yolo

def get_target_shape(img_shape, target_size: int):
    h, w = img_shape[:2]
    new_w, new_h = (int(target_size * w / h), target_size) if h > w else (target_size, int(target_size * h / w))
    return new_h, new_w

def _create_realesrgan_degradation_pipeline(img_shape, mosaic_size: int, device: str, p:float):
    target_h, target_w = img_shape[:2]

    if not np.random.uniform() < p:
        return torchvision_transforms.Resize(size=(target_h, target_w))

    sharpener = UnsharpMaskingSharpener().to(device)
    jpeger = DiffJPEG(differentiable=False).to(device)
    kernel_range = [2 * v + 1 for v in range(3, 5)]

    # TODO: adjust small mosaic logic, probably should pass this down from creation func
    small_mosaic_blocks = mosaic_size < min(img_shape[:2]) * 14 / 1000
    if small_mosaic_blocks:
        # skip heavy degradations
        first_pass = lambda img: img
    else:
        first_pass = torchvision_transforms.Compose([
            lada_transforms.Blur(kernel_range=kernel_range, kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'], kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                                 sinc_prob=0.1, blur_sigma=[0.2, 0.6], betag_range=[0.5, 4], betap_range=[1, 2], device=device, p=0.4),
            lada_transforms.Resize(resize_range=[0.75, 1.25], resize_prob=[0.2, 0.7, 0.1], target_base_h=target_h, target_base_w=target_w, p=0.8),
            lada_transforms.GaussianPoissonNoise(sigma_range=[0., 1.6], poisson_scale_range=[0., 0.3], gaussian_noise_prob=0.5, gray_noise_prob=0.4, p=0.8),
            lada_transforms.JPEGCompression(jpeger, jpeg_range=[45, 95], p=0.7),
        ])

    return torchvision_transforms.Compose([
        torchvision_transforms.Resize(size=(target_h, target_w)),
        lada_transforms.Sharpen(sharpener, p=0.5),
        first_pass,
        lada_transforms.Blur(kernel_range=kernel_range, kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'], kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                             sinc_prob=0.1, blur_sigma=[0.1, 0.6], betag_range=[0.5, 4], betap_range=[1, 2], device=device, p=0.4),
        lada_transforms.Resize(resize_range=[0.85, 1.15], resize_prob= [0.3, 0.4, 0.3], target_base_h=target_h, target_base_w=target_w, p=0.8),
        lada_transforms.GaussianPoissonNoise(sigma_range=[0., 1.6], poisson_scale_range=[0., 0.3], gaussian_noise_prob=0.5, gray_noise_prob=0.4, p=0.8),
        torchvision_transforms.RandomChoice(transforms=[
            torchvision_transforms.Compose([
                lada_transforms.Resize(resize_range=[1., 1.], resize_prob=[0, 0, 1], target_base_h=target_h, target_base_w=target_w, p=1.0),
                lada_transforms.SincFilter(kernel_range=kernel_range, sinc_prob=0., device=device, p=0.),
                lada_transforms.JPEGCompression(jpeger, jpeg_range=[65, 95], p=0.7),
            ]),
            torchvision_transforms.Compose([
                lada_transforms.JPEGCompression(jpeger, jpeg_range=[65, 95], p=0.7),
                lada_transforms.Resize(resize_range=[1., 1.], resize_prob=[0, 0, 1], target_base_h=target_h, target_base_w=target_w, p=1.0),
                lada_transforms.SincFilter(kernel_range=kernel_range, sinc_prob=0., device=device, p=0.),
            ])
        ], p=[0.5, 0.5]),
    ])

def create_degradation_pipeline(img_shape: tuple[int, int, int], mosaic_size: int, device='cuda'):
    return torchvision_transforms.Compose([
        lada_transforms.Image2Tensor(bgr2rgb=False, unsqueeze=True, device=device),
        _create_realesrgan_degradation_pipeline(img_shape, mosaic_size=mosaic_size, device=device, p=0.8),
        lada_transforms.Tensor2Image(rgb2bgr=False, squeeze=True),
        lada_transforms.VideoCompression(p=0.3, codecs=['libx264', 'libx265'], codec_probs=[0.5, 0.5],
                                         crf_ranges={'libx264': (26, 32), 'libx265': (28, 34)},
                                         bitrate_ranges={}),
    ])

def get_detections(source: str | Image, detectors: list[NsfwImageDetector | FaceDetector | HeadDetector]) -> Detections:
    detections = []
    nsfw_detections = []
    sfw_detections = []
    frame = None

    for detector in detectors:
        _detections = detector.detect(source)
        if _detections is None:
            continue
        if frame is None:
            frame = _detections.frame
        if isinstance(detector, NsfwImageDetector):
            nsfw_detections.extend(_detections.detections)
        else:
            sfw_detections.extend(_detections.detections)

    skip = []
    def get_non_skipped(detections):
        non_skipped = []
        for det in detections:
            skip_det = False
            for skipped_det in skip:
                if skipped_det is det:
                    skip_det = True
                    break
            if not skip_det:
                non_skipped.append(det)
        return non_skipped

    for sfw_detection in sfw_detections:
        should_skip = False
        for nsfw_detection in nsfw_detections:
            if box_overlap(sfw_detection.box, nsfw_detection.box):
                skip.append(sfw_detection)
                should_skip = True
                break
        if should_skip: continue
        for _sfw_detection in get_non_skipped(sfw_detections):
            if _sfw_detection is sfw_detection:
                continue
            if box_overlap(sfw_detection.box, _sfw_detection.box):
                skip.append(sfw_detection)
                should_skip = True
                break
        if should_skip: continue
        detections.append(sfw_detection)
    for nsfw_detection in nsfw_detections:
        should_skip = False
        for _nsfw_detection in get_non_skipped(nsfw_detections):
            if _nsfw_detection is nsfw_detection:
                continue
            if box_overlap(nsfw_detection.box, _nsfw_detection.box):
                skip.append(nsfw_detection)
                should_skip = True
                break
        if should_skip: continue
        detections.append(nsfw_detection)
    return Detections(frame, detections)

def wait_until_key_press(accepted_keys: list[str]) -> str:
    _accepted_keys = [ord(key) for key in accepted_keys]
    while True:
        key_pressed = cv2.waitKey(1)
        _key_pressed = key_pressed & 0xFF
        if _key_pressed in _accepted_keys:
            return accepted_keys[_accepted_keys.index(_key_pressed)]

def show_image_file(file_path, detectors: list[NsfwImageDetector | FaceDetector | HeadDetector], device='cpu', window_name="mosaic", target_size=640) -> bool:
    img = cv2.imread(file_path)
    target_shape = get_target_shape(img.shape, target_size)
    img = image_utils.resize(img, size=target_shape)

    detections: Detections = get_detections(img, detectors)
    if not detections or len(detections.detections) == 0:
        return True

    while True:
        mask, img_mosaic, mask_mosaic, mosaic_size = None, None, None, None
        for detection in detections.detections:
            if mask is None:
                mask = detection.mask
            else:
                mask = mask | detection.mask
            if img_mosaic is None:
                img_mosaic, mask_mosaic, mosaic_size = lada_transforms.Mosaic(reuse_input_mask_value=True)(img, detection.mask)
            else:
                img_mosaic, _mask_mosaic, _mosaic_size = lada_transforms.Mosaic(reuse_input_mask_value=True)(img_mosaic, detection.mask)
                mask_mosaic = mask_mosaic | _mask_mosaic
                mosaic_size = min(_mosaic_size, mosaic_size)

        degrade = create_degradation_pipeline(img.shape, device=device, mosaic_size=mosaic_size)

        degraded_mosaic = degrade(img_mosaic)
        mask_mosaic = image_utils.resize(mask_mosaic, degraded_mosaic.shape[:2], interpolation=cv2.INTER_NEAREST)

        show_img = visualization_utils.overlay_mask_boundary(degraded_mosaic, mask_mosaic, color=(0, 255, 0))
        mask = image_utils.resize(mask, degraded_mosaic.shape[:2], interpolation=cv2.INTER_NEAREST)
        show_img = visualization_utils.overlay_mask_boundary(show_img, mask, color=(255, 0, 0))

        cv2.imshow(window_name, show_img)
        pressed_key = wait_until_key_press(["n", "r", "q"])
        if pressed_key == "n":
            break
        elif pressed_key == "r":
            continue
        elif pressed_key == "q":
            return False
    return True

def process_image_file(file_path, output_root, detectors: list[NsfwImageDetector | FaceDetector | HeadDetector], device='cpu', target_size=640):
    img = cv2.imread(file_path)
    target_shape = get_target_shape(img.shape, target_size)
    img = image_utils.resize(img, size=target_shape)

    detections: Detections = get_detections(img, detectors)
    if not detections or len(detections.detections) == 0:
        name = osp.splitext(os.path.basename(file_path))[0]
        shutil.copy(file_path, f"{output_root}/background_images/{name}.jpg")
        return

    mask, img_mosaic, mask_mosaic, mosaic_size = None, None, None, None
    for detection in detections.detections:
        if mask is None:
            mask = detection.mask
        else:
            mask = mask | detection.mask
        if img_mosaic is None:
            img_mosaic, mask_mosaic, mosaic_size = lada_transforms.Mosaic(reuse_input_mask_value=True)(img, detection.mask)
        else:
            img_mosaic, _mask_mosaic, _mosaic_size = lada_transforms.Mosaic(reuse_input_mask_value=True)(img_mosaic, detection.mask)
            mask_mosaic = mask_mosaic | _mask_mosaic
            mosaic_size = min(_mosaic_size, mosaic_size)

    degrade = create_degradation_pipeline(img.shape, device=device, mosaic_size=mosaic_size)

    degraded_mosaic = degrade(img_mosaic)
    mask_mosaic = image_utils.resize(mask_mosaic, degraded_mosaic.shape[:2], interpolation=cv2.INTER_NEAREST)

    name = osp.splitext(os.path.basename(file_path))[0]
    cv2.imwrite(f"{output_root}/images/{name}.jpg", degraded_mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(f"{output_root}/images_hq/{name}.jpg", img_mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(f"{output_root}/masks/{name}.png", mask_mosaic)

def get_files(dir, filter_func):
    file_list = []
    for r, d, f in os.walk(dir):
        for file in f:
            file_path = osp.join(r, file)
            if filter_func(file_path):
                file_list.append(Path(file_path))
    return file_list

def parse_args():
    parser = argparse.ArgumentParser("Create mosaic detection dataset")
    parser.add_argument('--output-root', type=Path, help="directory where resulting images/masks are saved")
    parser.add_argument('--input-root', type=Path, help="directory containing image files")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--model', type=str, default="model_weights/lada_nsfw_detection_model_v1.3.pt", help="path to NSFW detection model")
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads")
    parser.add_argument('--start-index', type=int, default=0, help="Can be used to continue a previous run. Note the index number next to last processed file name")
    parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help="show each sample")
    parser.add_argument('--create-nsfw-mosaics', default=True, action=argparse.BooleanOptionalAction, help="Use Lada NSFW detection model to create NSFW mosaics")
    parser.add_argument('--create-sfw-face-mosaics', default=False, action=argparse.BooleanOptionalAction, help="Use CenterFace human face detection model to create SFW mosaics")
    parser.add_argument('--create-sfw-head-mosaics', default=False, action=argparse.BooleanOptionalAction, help="Use BPJDet human head detection model to create SFW mosaics")
    parser.add_argument('--max-file-limit', type=int, default=None, help="instead of processing all files found in input-root dir it will choose files randomly up to the given limit")
    parser.add_argument('--target-size', type=int, default=640, help="output size of images. should match imgsz param of YOLO")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    detectors = []
    if args.create_nsfw_mosaics:
        model = Yolo(args.model)
        detectors.append(NsfwImageDetector(model, args.device, random_extend_masks=True, conf=0.8))
    if args.create_sfw_face_mosaics:
        model = CenterFace()
        detectors.append(FaceDetector(model, random_extend_masks=True, conf=0.8))
    if args.create_sfw_head_mosaics:
        model = bpjdet.get_model(device=args.device)
        data = bpjdet.JointBP_CrowdHuman_head.DATA
        data['conf_thres_part'] = 0.7
        data['iou_thres_part'] = 0.7
        data['match_iou_thres'] = 0.7
        detectors.append(HeadDetector(model, data=data, random_extend_masks=True, conf_thres=data['conf_thres_part'], iou_thres=data['iou_thres_part']))
    assert len(detectors) > 0

    selected_files = get_files(args.input_root, image_utils.is_image_file)
    if args.max_file_limit and len(selected_files) > args.max_file_limit:
        selected_files = random.choices(selected_files, k=args.max_file_limit)

    if args.show:
        for file_idx, file_path in enumerate(selected_files):
            print(f"{file_idx}, Processing {file_path.name}")
            should_continue = show_image_file(file_path, detectors, device=args.device, target_size=args.target_size)
            if not should_continue: break
        cv2.destroyAllWindows()
    else:
        os.makedirs(f"{args.output_root}/masks", exist_ok=True)
        os.makedirs(f"{args.output_root}/images", exist_ok=True)
        os.makedirs(f"{args.output_root}/images_hq", exist_ok=True)
        os.makedirs(f"{args.output_root}/background_images", exist_ok=True)
        os.makedirs(f"{args.output_root}/detection_labels", exist_ok=True)
        os.makedirs(f"{args.output_root}/segmentation_labels", exist_ok=True)
        jobs = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for file_idx, file_path in enumerate(selected_files):
                if file_idx < args.start_index or len(list(args.output_root.glob(f"*/{file_path.name}*"))) > 0:
                    print(f"{file_idx}, Skipping {file_path.name}: Already processed")
                    continue
                print(f"{file_idx}, Processing {file_path.name}")
                jobs.append(executor.submit(process_image_file, file_path, args.output_root, detectors, args.device, args.target_size))
                clean_up_completed_futures(jobs)
        wait(jobs, return_when=ALL_COMPLETED)
        clean_up_completed_futures(jobs)

        print("Finished processing images. Now converting dataset to YOLO format...")

        # Remap to classes we want to train with. Must match mosaic_detection_dataset_config.yaml
        pixel_to_class_mapping = {
            DETECTION_CLASSES["nsfw"]["mask_value"]: 0,
            DETECTION_CLASSES["sfw_face"]["mask_value"]: 1,
            DETECTION_CLASSES["sfw_head"]["mask_value"]: 1
        }
        convert_segment_masks_to_yolo_labels(f"{args.output_root}/masks", f"{args.output_root}/segmentation_labels", f"{args.output_root}/detection_labels", pixel_to_class_mapping)

if __name__ == '__main__':
    main()