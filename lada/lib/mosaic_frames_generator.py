from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from lada.lib import Box, Mask, Image, VideoMetadata
from lada.lib import image_utils
from lada.lib import visualization
from lada.lib.scene_utils import crop_to_box_v3
from lada.lib.clean_frames_generator import convert_yolo_box, convert_yolo_mask
from lada.lib import video_utils


class Scene:
    def __init__(self, file_path: Path, video_meta_data: VideoMetadata):
        self.file_path = file_path
        self.video_meta_data = video_meta_data
        self.data: list = []
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0

    def __len__(self):
        return len(self.data)

    def add_frame(self, frame_num: int, img: Image, mask: Mask, box: Box):
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
            self.data.append((img, mask, box))
        else:
            assert frame_num == self.frame_end + 1
            self.frame_end = frame_num
            self.data.append((img, mask, box))

    def merge_mask_box(self, mask: Mask, box: Box):
        assert self.belongs(box)
        current_box = self.data[-1][2]
        t = min(current_box[0], box[0])
        l = min(current_box[1], box[1])
        b = max(current_box[2], box[2])
        r = max(current_box[3], box[3])
        new_box = (t, l, b, r)

        current_mask = self.data[-1][1]
        new_mask = np.maximum(current_mask, mask)

        self.data[-1] = self.data[-1][0], new_mask, new_box

    def get_images(self):
        return [img for img, _, _ in self.data]

    def get_masks(self):
        return [mask for _, mask, _ in self.data]

    def get_boxes(self):
        return [box for _, _, box in self.data]

    def belongs(self, box: Box):
        if len(self.data) == 0:
            return False
        last_scene_box = self.data[-1][2]
        return box_overlaps(last_scene_box, box)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


class Clip:
    def __init__(self, scene: Scene, size, pad_mode, id, preserve_relative_scale):
        self.id = id
        self.file_path = scene.file_path
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        assert self.frame_start <= self.frame_end
        self.size = size
        self.pad_mode = pad_mode
        self.data = []
        self._index: int = 0
        scene_masks = scene.get_masks()
        scene_images = scene.get_images()
        scene_boxes = scene.get_boxes()
        pad_before_resize = (0,0,0,0)
        pad_after_resize = (0, 0, 0, 0)

        # crop scene
        for i in range(len(scene)):
            img, mask, box = scene_images[i], scene_masks[i], scene_boxes[i]
            cropped_img, cropped_mask, cropped_box, _ = crop_to_box_v3(box, img, mask, (size, size), max_box_expansion_factor=1., border_size=0.06)
            self.data.append((cropped_img, cropped_mask, cropped_box, cropped_img.shape, pad_after_resize, pad_before_resize))

        # resize crops to out_size
        if preserve_relative_scale:
            max_width, max_height = self.get_max_width_height()

        for i, (cropped_img, cropped_mask, cropped_box, _, _, _) in enumerate(self.data):
            if preserve_relative_scale:
                cropped_img, pad_before_resize = image_utils.pad_image(cropped_img, max_height, max_width, mode=self.pad_mode)
                cropped_mask, _ = image_utils.pad_image(cropped_mask, max_height, max_width, mode='zero')

            crop_shape = cropped_img.shape

            cropped_img = image_utils.resize(cropped_img, size, interpolation=cv2.INTER_LINEAR)
            cropped_mask = image_utils.resize(cropped_mask, size, interpolation=cv2.INTER_NEAREST)

            cropped_img, pad_after_resize = image_utils.pad_image(cropped_img, size, size, mode=self.pad_mode)
            cropped_mask, _ = image_utils.pad_image(cropped_mask, size, size, mode='zero')

            self.data[i] = (cropped_img, cropped_mask, cropped_box, crop_shape, pad_after_resize, pad_before_resize)

    def get_max_width_height(self):
        max_width = 0
        max_height = 0
        for box in self.get_clip_boxes():
            t, l, b, r = box
            width, height = r - l + 1, b - t + 1
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        return max_width, max_height

    def get_clip_images(self):
        return [clip_img for clip_img, _, _, _, _, _ in self.data]

    def get_clip_boxes(self):
        return [clip_box for _, _, clip_box, _, _, _ in self.data]

    def pop(self):
        self.frame_start += 1
        if self.frame_start > self.frame_end:
            self.frame_start = None
            self.frame_end = None
        return self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.data[item]

def box_overlaps(box1: Box, box2: Box) -> bool:
    y_overlaps = (box1[0] <= box2[0] <= box1[2] or box1[0] <= box2[2] <= box1[2]) or (box2[0] <= box1[0] <= box2[2] or box2[0] <= box1[2] <= box2[2])
    x_overlaps = (box1[1] <= box2[1] <= box1[3] or box1[1] <= box2[3] <= box1[3]) or (box2[1] <= box1[1] <= box2[3] or box2[1] <= box1[3] <= box2[3])
    return y_overlaps and x_overlaps

class MosaicFramesGenerator:
    def __init__(self, model: ultralytics.models.YOLO, video_file, max_clip_length=30, clip_size=256, device=None, pad_mode='reflect', preserve_relative_scale=False, dont_preserve_relative_scale=False, start_ns=0):
        self.model = model
        self.video_file = video_file
        self.device = torch.device(device) if device is not None else device
        self.max_clip_length = max_clip_length
        self.clip_size = clip_size
        self.preserve_relative_scale = preserve_relative_scale
        self.dont_preserve_relative_scale = dont_preserve_relative_scale
        self.pad_mode = pad_mode
        self.clip_counter = 0
        self.start_ns = start_ns
        self.video_meta_data = video_utils.get_video_meta_data(self.video_file)
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)

        assert preserve_relative_scale or dont_preserve_relative_scale

    def __call__(self, *args, **kwargs) -> Generator[Clip, None, None]:
        with video_utils.VideoReader(self.video_file) as video_reader:
            video_frames_generator = video_reader.frames()
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)
            scenes: list[Scene] = []
            for frame_num in range(self.start_frame, self.video_meta_data.frames_count):
                frame, _ = next(video_frames_generator)
                for results in self.model.predict(source=frame, stream=False, verbose=False, device=self.device):
                    for i in range(len(results.boxes)):
                        mask = convert_yolo_mask(results.masks[i], results.orig_shape)
                        box = convert_yolo_box(results.boxes[i], results.orig_shape)

                        current_scene = None
                        for scene in scenes:
                            if scene.belongs(box):
                                if scene.frame_end == frame_num:
                                    current_scene = scene
                                    current_scene.merge_mask_box(mask, box)
                                else:
                                    current_scene = scene
                                    current_scene.add_frame(frame_num, results.orig_img, mask, box)
                                break
                        if current_scene is None:
                            current_scene = Scene(self.video_file, self.video_meta_data)
                            scenes.append(current_scene)
                            current_scene.add_frame(frame_num, results.orig_img, mask, box)

                    completed_scenes = []
                    for current_scene in scenes:
                        if (current_scene.frame_end < frame_num or len(current_scene) >= self.max_clip_length or current_scene.frame_end == (self.video_meta_data.frames_count - 1)) and current_scene not in completed_scenes:
                            completed_scenes.append(current_scene)
                            other_scenes = [other for other in scenes if other != current_scene]
                            for other_scene in other_scenes:
                                if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                                    completed_scenes.append(other_scene)

                    for completed_scene in sorted(completed_scenes, key=lambda s: s.frame_start):
                        if self.preserve_relative_scale and self.dont_preserve_relative_scale:
                            clip_v1 = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter, True)
                            clip_v2 = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter, False)
                            yield clip_v1, clip_v2
                        elif self.preserve_relative_scale:
                            clip = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter, True)
                            yield clip
                        elif self.dont_preserve_relative_scale:
                            clip = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter, False)
                            yield clip
                        #print(f"frame {frame_num}, yielding clip starting {clip.frame_start}, ending {clip.frame_end}, all scene starts: {[s.frame_start for s in scenes]}, completed scenes: {[s.frame_start for s in completed_scenes]}")
                        scenes.remove(completed_scene)
                        self.clip_counter += 1

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str)
        parser.add_argument('--model-path', type=str,
                            default='yolo/runs/segment/train_mosaic_detection_yolov9c/weights/best.pt')
        parser.add_argument('--max-clip-length', type=int, default=30)
        parser.add_argument('--clip-size', type=int, default=256)

        args = parser.parse_args()
        return args

    def show(window_name, output):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, output)
        while True:
            key_pressed = cv2.waitKey(1)
            if key_pressed & 0xFF == ord("n"):
                break

    args = parse_args()

    model = YOLO(args.model_path)

    mosaic_generator = MosaicFramesGenerator(model, args.input, args.max_clip_length, args.clip_size, should_pad=False)

    window_name = "main"

    clip_buffer = []
    frame_buffer = []

    clip_colors = {}
    clip_names = {}

    cap = cv2.VideoCapture(args.input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(frame_num):
        ret, frame = cap.read()
        if ret:
            visualization.draw_text(f"frame:{frame_num}", (25,25), frame)
            return frame
        return None

    frame_num = 0
    clips_min_frame_start = 0

    for clip_idx, clip in enumerate(mosaic_generator()):
        if clip.frame_start > frame_num:
            if len(clip_buffer) == 0:
                while clip.frame_start > frame_num:
                    frame = read_frame(frame_num)
                    show(window_name, frame)
                    frame_num += 1
            else:
                while True:
                    if len(clip_buffer) == 0 or clip.frame_start <= min(clip_buffer, key=lambda c: c.frame_start).frame_start:
                        break
                    frame = read_frame(frame_num)
                    for buffered_clip in [c for c in clip_buffer if c.frame_start == frame_num]:
                        clip_start_frame_num = buffered_clip.frame_start
                        clip_img, clip_mask, clip_box, orig_crop_shape, pad = buffered_clip.pop()
                        visualization.draw_text(f"c:{clip_names[buffered_clip]},f:{clip_start_frame_num}", (25, 25), clip_img)
                        t, l, b, r = clip_box
                        frame[t:b + 1, l:r + 1, :] = image_utils.resize(clip_img, orig_crop_shape[:2])
                        visualization.draw_box(frame, clip_box, color=clip_colors[buffered_clip])
                    show(window_name, frame)
                    frame_num += 1

                    processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
                    for processed_clip in processed_clips:
                        clip_buffer.remove(processed_clip)
                        del clip_colors[processed_clip]
                        del clip_names[processed_clip]

        clip_buffer.append(clip)
        clip_colors[clip] = list(np.random.random(size=3) * 256)
        clip_names[clip] = clip_idx

    while frame_num < frame_count:
        if len(clip_buffer) == 0:
            frame = read_frame(frame_num)
            show(window_name, frame)
            frame_num += 1
        else:
            while True:
                if len(clip_buffer) == 0:
                    break
                frame = read_frame(frame_num)
                for buffered_clip in [c for c in clip_buffer if c.frame_start == frame_num]:
                    clip_start_frame_num = buffered_clip.frame_start
                    clip_img, clip_mask, clip_box, orig_crop_shape, pad = buffered_clip.pop()
                    visualization.draw_text(f"c:{clip_names[buffered_clip]},f:{clip_start_frame_num}", (25, 25), clip_img)
                    t, l, b, r = clip_box
                    frame[t:b + 1, l:r + 1, :] = image_utils.resize(clip_img, orig_crop_shape[:2])
                    visualization.draw_box(frame, clip_box, color=clip_colors[buffered_clip])
                show(window_name, frame)
                frame_num += 1

                processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
                for processed_clip in processed_clips:
                    clip_buffer.remove(processed_clip)
                    del clip_colors[processed_clip]
                    del clip_names[processed_clip]

    cap.release()
    cv2.destroyAllWindows()
