import glob
import os
from pathlib import Path

import numpy as np
import torch
import argparse
import cv2

from lada import MODEL_WEIGHTS_DIR
from lada.bpjdet.utils.augmentations import letterbox
from lada.bpjdet.utils.general import check_img_size, non_max_suppression, scale_coords
from lada.bpjdet.models.experimental import attempt_load
from lada.bpjdet.data import JointBP_CrowdHuman_head
from lada.lib import Box

here = Path(__file__).parent.resolve()

_colors_list = [
        # [255, 0, 0], [255, 127, 0], [255, 255, 0], [127, 255, 0], [0, 255, 0], [0, 255, 127], 
        # [0, 255, 255], [0, 127, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127],
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255],
        [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127],
        [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0],
        [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
    ]  # 27 colors

def _load_image(file_path:str, device: torch.device, imgz: int, stride: int) -> torch.tensor:
    img_orig = cv2.imread(file_path)  # BGR
    img = letterbox(img_orig, imgz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    return img_orig, img

def _cal_inside_iou(bigBox, smallBox):  # body_box, part_box
    # calculate small rectangle inside big box ratio, calSmallBoxInsideRatio
    [Ax0, Ay0, Ax1, Ay1] = bigBox[0:4]
    [Bx0, By0, Bx1, By1] = smallBox[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        # return crossArea/(areaA + areaB - crossArea)
        return crossArea/areaB  # range [0, 1]

def _post_process_batch(data, imgs, paths, shapes, body_dets, part_dets):

    batch_bboxes, batch_points, batch_scores, batch_imgids = [], [], [], []
    batch_parts_dict = {}
    img_indexs = []

    # process each image in batch
    for si, (bdet, pdet) in enumerate(zip(body_dets, part_dets)):
        nbody, npart = bdet.shape[0], pdet.shape[0]

        if nbody:  # one batch
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]

            img_id = int(os.path.splitext(os.path.split(path)[-1])[0]) if path else si

            scores = bdet[:, 4].cpu().numpy()  # body detection score
            bboxes = scale_coords(imgs[si].shape[1:], bdet[:, :4], shape).round().cpu().numpy()
            points = scale_coords(imgs[si].shape[1:], bdet[:, -data['num_offsets']:], shape).cpu().numpy()
            points = points.reshape((nbody, -1, 2))
            # points = np.concatenate((points, np.zeros((nbody, points.shape[1], 1))), axis=-1)  # n*c*2 --> n*c*3
            points = np.concatenate((points, np.zeros((nbody, points.shape[1], 5))), axis=-1)  # n*c*2 --> n*c*7

            batch_parts_dict[str(img_id)] = []
            if npart:
                pdet[:, :4] = scale_coords(imgs[si].shape[1:], pdet[:, :4].clone(), shape)
                pdet_slim = pdet[:, :6].cpu()

                matched_part_ids = [-1 for i in range(points.shape[0])]  # points shape is n*c*7, add in 2022-12-09
                for id, (x1, y1, x2, y2, conf, cls) in enumerate(pdet_slim):
                    p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))  # the body-part's part bbox center point
                    part_pts = points[:, int(cls - 1)]
                    dist = np.linalg.norm(part_pts[:, :2] - np.array([[p_xc, p_yc]]), axis=-1)
                    pt_match = np.argmin(dist)

                    tmp_iou = _cal_inside_iou(bboxes[pt_match], [x1, y1, x2, y2])  # add in 2022-12-11, body-part must inside the body
                    if conf > part_pts[pt_match][2] and tmp_iou > data['match_iou_thres']:  # add in 2022-12-09, we fetch the part bbox with highest conf
                        part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        matched_part_ids[pt_match] = id

                    # put all detected body part bboxes into their image_dict
                    batch_parts_dict[str(img_id)].append([x1, y1, x2, y2, conf, cls])


            batch_bboxes.extend(bboxes)
            batch_points.extend(points)
            batch_scores.extend(scores)
            batch_imgids.extend([img_id] * len(scores))

            img_indexs.append(si)
        else:
            pass
            #print("This image has no object detected!")

    return batch_bboxes, batch_points, batch_scores, batch_imgids, batch_parts_dict, img_indexs

def get_model(device: str, weights_path: str):
    torch_device = torch.device(device)
    return attempt_load(weights_path, map_location=torch_device)

def inference(model, image_path, imgz, data, conf_thres=0.45, iou_thres=0.75) -> list[Box]:
    stride = int(model.stride.max())
    imgz = check_img_size(imgz, s=stride)
    device = next(model.parameters()).device

    img_orig, img = _load_image(image_path, device=device, stride=stride, imgz=imgz)

    scales = [1]
    batch = torch.unsqueeze(img, 0)  # expand for batch dim
    out_ori = model(batch, augment=True, scales=scales)[0]
    body_dets = non_max_suppression(out_ori, conf_thres, iou_thres,
                                    classes=[0], num_offsets=data['num_offsets'])
    part_dets = non_max_suppression(out_ori, conf_thres, iou_thres,
                                    classes=list(range(1, 1 + data['num_offsets'] // 2)), num_offsets=data['num_offsets'])

    # Post-processing of body and part detections
    bboxes, points, scores, _imgids, _parts_dict, _img_indexs = _post_process_batch(data, batch, [], [[img_orig.shape[:2]]], body_dets, part_dets)

    head_detections = []
    for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
        f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
        if f_score == 0:  # for the body-head pair, we must have a detected head
            continue
        [px1, py1, px2, py2] = f_bbox
        head_detection_box: Box = int(py1), int(px1), int(py2), int(px2)
        head_detections.append(head_detection_box)
    return head_detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--imgsz', type=int, default=1536)
    parser.add_argument('--weights', default=os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'ch_head_s_1536_e150_best_mMR.pt'))
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.75, help='NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.6, help='Matching IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--line-thick', type=int, default=2, help='thickness of lines')

    args = parser.parse_args()


    device = torch.device(args.device)

    model = get_model(args.device, args.weights)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size

    if os.path.isdir(args.img_path):
        files = sorted(glob.glob(os.path.join(args.img_path, '*.*')))
    elif os.path.isfile(args.img_path):
        files = [args.img_path]
    
    data = JointBP_CrowdHuman_head.DATA
    data['conf_thres_part'] = args.conf_thres  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = args.iou_thres  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = args.match_iou  # whether a body-part in matched with one body bbox
    
    print(args.img_path, len(files))
    for index, single_path in enumerate(files):
        
        im0 = cv2.imread(single_path)

        detections = inference(model, single_path, data=data, imgz=args.imgsz, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

        print(index, single_path, "\n")

        args.line_thick = max(im0.shape[:2]) // 1000 + 3
        
        for i, box in enumerate(detections):
            [py1, px1, py2, px2] = box
            
            color = _colors_list[i % len(_colors_list)]

            cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick) # draw head

        if len(detections) > 0:
            cv2.imwrite(single_path[:-4]+"_res_head.jpg", im0)

            
        
  