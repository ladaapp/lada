# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import hashlib
import os

import cv2
import numpy as np

from lada.models.yolo.yolo import Yolo
from lada.utils import visualization_utils

class InferenceViewer:
    def __init__(self, model_path: str, input_file_path: str, conf: float, imgsz: int, iou: float, screenshots_dir: str | None):
        self.model: Yolo = Yolo(model_path)
        self.input: str = input_file_path
        self.imgsz: int = imgsz
        self.iou: float = iou
        self.conf: float = conf
        self.screenshots_dir: str = screenshots_dir
        self.is_video_file: bool = True
        self.frame_num: int = 0
        self.window_name: str = 'img'
        self.vid_capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray[np.uint8] | None = None
        self.imgsz_max: int = 1920
        self.imgsz_min: int = 384
        self.conf_min: float = 0.001
        self.iou_min: float = 0.3

    def load_input(self):
        try:
            self.frame = cv2.imread(self.input)
            if self.frame is None:
                self.is_video_file = True
                self.vid_capture = cv2.VideoCapture(self.input)
                if not self.vid_capture.isOpened():
                    raise Exception("Unable to read from input file")
                ret, self.frame = self.vid_capture.read()
            else:
                self.is_video_file = False
        except Exception as e:
            print(f"Error loading input: {e}")
            exit(1)

    def setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.is_video_file:
            frame_count = int(self.vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            cv2.createTrackbar('frame', self.window_name, 0, frame_count, self.update_frame_num)
        cv2.createTrackbar('conf', self.window_name, int(self.conf * 100), 100, self.update_conf)
        cv2.createTrackbar('imgsz', self.window_name, int(self.imgsz/32), int((self.imgsz_max-self.imgsz_min)/32), self.update_imgsz)
        cv2.createTrackbar('iou', self.window_name, int(self.iou * 100 / 5), 20, self.update_iou)

    def update(self):
        if self.is_video_file:
            self.vid_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
            ret, frame = self.vid_capture.read()
            if ret:
                self.frame = frame

        result = self.model.predict(self.frame, conf=self.conf, imgsz=self.imgsz, iou=self.iou)
        output = result[0].plot()
        cv2.imshow(self.window_name, output)

    def update_frame_num(self, frame_num):
        self.frame_num = frame_num
        self.update()

    def update_conf(self, conf):
        self.conf = max(self.conf_min, conf / 100.)
        print("conf", self.conf)
        self.update()

    def update_iou(self, iou: float):
        self.iou = max(self.iou_min, (iou * 5) / 100.)
        print("iou", iou, self.iou)
        self.update()

    def update_imgsz(self, imgsz: int):
        self.imgsz = imgsz * 32 + self.imgsz_min
        print("imgsz", imgsz, self.imgsz)
        self.update()

    def screenshot(self, dir):
        if dir is None:
            print("Cannot save screenshot as no screenshot directory was specified")
            return
        os.makedirs(dir, exist_ok=True)

        file_path = os.path.join(dir, f"{hashlib.sha256(self.frame).hexdigest()}-{self.frame_num}.jpg")
        print(f"Saved screenshot of frame {self.frame_num} as {file_path}")
        cv2.imwrite(file_path, self.frame)

    def run(self):
        try:
            self.load_input()
            self.setup_window()
            self.update()

            while True:
                pressed_key = visualization_utils.wait_until_key_press(["q", "s"])
                if pressed_key == "s":
                    self.screenshot(self.screenshots_dir)
                elif pressed_key == "q":
                    print("Exiting...")
                    break

        except KeyboardInterrupt:
            print("Closed with Ctrl+C")

        finally:
            # Clean up resources
            cv2.destroyAllWindows()
            if self.is_video_file:
                self.vid_capture.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to image or video file")
    parser.add_argument('--model-path', type=str, required=True, help="Path to YOLO model weights file")
    parser.add_argument('--conf', type=float, default=0.4, help="Default confidence threshold. Can be adjusted interactively")
    parser.add_argument('--iou', type=float, default=0.7, help="Default IoU (Intersection over union) used for NMS (Non-Maximum-Suppression). Can be adjusted interactively")
    parser.add_argument('--imgsz', type=int, default=640, help="Default Target Image/Frame resolution. Image/Frame will be scaled up/down accordingly. Needs to be a multiple of 32 to match YOLO stride size. Can be adjusted interactively")
    parser.add_argument('--screenshot-dir', type=str, default=None, help="Path to store screenshots. Save the current frame by pressing 's'")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.imgsz % 32 == 0, "Incompatible imgsz. Must be compatible with YOLO stride so a multiple of 32"
    viewer = InferenceViewer(args.model_path, args.input, args.conf, args.imgsz, args.iou, args.screenshot_dir)
    viewer.run()
