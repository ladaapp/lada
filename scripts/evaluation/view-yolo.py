# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import os
import hashlib
from ultralytics import YOLO
import argparse


class InferenceViewer:
    def __init__(self, args):
        self.args = args
        self.net = YOLO(args.model_path)
        self.videomode = True
        self.frame_num = 0
        self.window_name = 'img'
        self.vid_capture = None
        self.frame = None

    def load_input(self):
        try:
            self.frame = cv2.imread(self.args.input)
            if self.frame is None:
                # input is a video file
                self.vid_capture = cv2.VideoCapture(self.args.input)
                if not self.vid_capture.isOpened():
                    raise Exception("Unable to read from input file")
                ret, self.frame = self.vid_capture.read()
            else:
                # input is an image file
                self.videomode = False
        except Exception as e:
            print(f"Error loading input: {e}")
            exit(1)

    def setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.videomode:
            frame_count = int(self.vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            cv2.createTrackbar('frame', self.window_name, 0, frame_count, self.update_frame)
        cv2.createTrackbar('thresh', self.window_name, int(self.args.mask_threshold * 100), 100, self.update_thresh)

    def update(self):
        if self.videomode:
            self.vid_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
            ret, frame = self.vid_capture.read()
            if ret:
                self.frame = frame

        result = self.net.predict(self.frame, conf=self.args.mask_threshold, imgsz=640)
        output = result[0].plot()
        cv2.imshow(self.window_name, output)

    def update_frame(self, f):
        self.frame_num = f
        self.update()

    def update_thresh(self, t):
        self.args.mask_threshold = t / 100.
        self.update()

    def screenshot(self, dir):
        if dir is None:
            return
        os.makedirs(dir, exist_ok=True)

        file_path = os.path.join(dir, f"{hashlib.sha256(self.frame).hexdigest()}-{self.frame_num}.jpg")
        print("Saved screenshot:", file_path)
        cv2.imwrite(file_path, self.frame)

    def run(self):
        try:
            self.load_input()
            self.setup_window()
            self.update()

            while True:
                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == ord("q"):
                    print("Exiting...")
                    break
                elif key_pressed == ord("s"):
                    print(f"Saving screenshot at frame {self.frame_num}")
                    self.screenshot(self.args.screenshot_dir)

        except KeyboardInterrupt:
            print("Closed with Ctrl+C")

        finally:
            # Clean up resources
            cv2.destroyAllWindows()
            if self.videomode:
                self.vid_capture.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--mask-threshold', type=float, default=0.4)
    parser.add_argument('--screenshot-dir', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    viewer = InferenceViewer(args)
    viewer.run()
