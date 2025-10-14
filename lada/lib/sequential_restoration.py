from __future__ import annotations

import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
DEVICE=0
os.environ['CUDA_VISIBLE_DEVICES'] = f'{DEVICE}'
import torch
import pycuda.driver as cuda
print(torch.cuda.is_available())
torch.jit.enable_onednn_fusion(True)
torch.set_float32_matmul_precision('high')
cuda_ctx = cuda.Device(DEVICE).retain_primary_context()
model_stream = torch.cuda.Stream(device=DEVICE)

import logging
from dataclasses import dataclass, field
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

from lada.lib.video_utils import VideoReader, get_video_meta_data
from lada.lib.mosaic_detection_model import MosaicDetectionModel
from lada.lib.mosaic_detector import Scene, Clip  # reuse data structs
from lada.lib import mask_utils, image_utils
from lada.lib.ultralytics_utils import convert_yolo_box, convert_yolo_mask
from lada.lib import visualization_utils
from lada.basicvsrpp.inference import inference as basicvsrpp_inference
from lada.deepmosaics.inference import restore_video_frames as deepmosaics_inference
from lada.deepmosaics.models import model_util as deepmosaics_model_util
from lada.lib.frame_restorer import load_models  # to mirror model loading if needed externally
from lada.lib.video_utils import VideoWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper data structures for storing per-frame restored patches
# ---------------------------------------------------------------------------
@dataclass
class Patch:
    img: np.ndarray          # restored RGB/BGR patch (same color space as frame bgr)
    mask: np.ndarray         # binary mask (same height/width as img) after resize/unpad
    box: Tuple[int, int, int, int]  # (t,l,b,r) in original frame coordinates

# Mapping: frame_number -> list[Patch]
FramePatches = Dict[int, List[Patch]]

import math
from tqdm import tqdm

class StreamingSequentialRestorer:
    def __init__(
        self,
        video_file: str,
        detection_model: MosaicDetectionModel,
        restoration_model,
        restoration_model_name: str,
        device: str,
        max_clip_length: int = 180,
        clip_size: int = 256,
        pad_mode: str = 'zero',
        batch_size: int = 4,
        detection_only: bool = False,
    ):
        self.video_file = video_file
        self.video_metadata = get_video_meta_data(video_file)
        self.detection_model = detection_model
        self.restoration_model = restoration_model
        self.restoration_model_name = restoration_model_name
        self.device = device
        self.max_clip_length = max_clip_length
        self.clip_size = clip_size
        self.pad_mode = pad_mode
        self.batch_size = batch_size
        self.detection_only = detection_only

        # Per-frame bookkeeping
        self.frames_buffer: dict[int, torch.Tensor] = {}
        self.mosaic_detected_flags: dict[int, bool] = {}
        self.frame_patches: FramePatches = {}
        self.next_frame_to_encode = 0
        self.frame_pts: dict[int, int | None] = {}

    # --------------- Internal helpers (reuse from SequentialRestorer) ------
    def _restore_clip_frames(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if self.detection_only:
            # visual overlay fallback: if util not present, return originals
            if hasattr(visualization_utils, 'draw_mosaic_detections_for_images'):
                return visualization_utils.draw_mosaic_detections_for_images(images)
            return images
        if self.restoration_model_name.startswith("deepmosaics"):
            return deepmosaics_inference(deepmosaics_model_util.device_to_gpu_id(self.device), self.restoration_model, images)
        elif self.restoration_model_name.startswith("basicvsrpp"):
            return basicvsrpp_inference(self.restoration_model, images, self.device)
        else:
            raise NotImplementedError()

    def _restore_clip_and_store_patches(self, clip: Clip):
        images = clip.get_clip_images()
        restored_images = self._restore_clip_frames(images) if not self.detection_only else images
        for i in range(len(restored_images)):
            img_rest = restored_images[i]
            orig_img, orig_mask, orig_box, orig_crop_shape, pad_after_resize = clip.data[i]
            clip.data[i] = img_rest, orig_mask, orig_box, orig_crop_shape, pad_after_resize

        frame_num = clip.frame_start
        for frame_index in range(len(clip)):
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = clip.data[frame_index]
            clip_img_unpadded = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask_unpadded = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img_resized = image_utils.resize(clip_img_unpadded, orig_crop_shape[:2])
            clip_mask_resized = image_utils.resize(clip_mask_unpadded, orig_crop_shape[:2], interpolation=cv2.INTER_NEAREST)
            t, l, b, r = orig_clip_box
            blend_mask = mask_utils.create_blend_mask(clip_mask_resized)
            patch = Patch(img=clip_img_resized, mask=blend_mask, box=(t, l, b, r))
            self.frame_patches.setdefault(frame_num + frame_index, []).append(patch)

    def _update_scenes_with_results(self, results, scenes: List[Scene], frame_num: int):
        has_mosaic = len(results.boxes) > 0
        self.mosaic_detected_flags[frame_num] = has_mosaic
        if not has_mosaic:
            return
        for i in range(len(results.boxes)):
            if self.detection_model.is_segmentation_model and results.masks is not None:
                mask = convert_yolo_mask(results.masks[i], results.orig_shape)
            else:
                mask = np.zeros(results.orig_shape, dtype=np.uint8)
            box = convert_yolo_box(results.boxes[i], results.orig_shape)
            current_scene = None
            for scene in scenes:
                if scene.belongs(box):
                    if scene.frame_end == frame_num:
                        scene.merge_mask_box(mask, box)
                    else:
                        scene.add_frame(frame_num, results.orig_img, mask, box)
                    current_scene = scene
                    break
            if current_scene is None:
                scene = Scene(self.video_file, self.video_metadata)
                scenes.append(scene)
                scene.add_frame(frame_num, results.orig_img, mask, box)

    def _flush_completed_scenes(self, scenes: List[Scene], frame_num: int, eof: bool, clip_counter_ref: List[int]):
        completed = []
        for sc in scenes:
            if (sc.frame_end < frame_num or len(sc) >= self.max_clip_length or eof) and sc not in completed:
                completed.append(sc)
                for other in scenes:
                    if other != sc and other.frame_start < sc.frame_start and other not in completed:
                        if other.frame_end < frame_num or len(other) >= self.max_clip_length or eof:
                            completed.append(other)
        for sc in sorted(completed, key=lambda s: s.frame_start):
            clip = Clip(sc, self.clip_size, self.pad_mode, clip_counter_ref[0])
            clip_counter_ref[0] += 1
            self._restore_clip_and_store_patches(clip)
            scenes.remove(sc)

    def _active_scene_min_start(self, scenes: List[Scene]) -> int:
        if not scenes:
            return math.inf
        return min(s.frame_start for s in scenes if s.frame_start is not None)

    def _frame_waiting_on_active_scene(self, frame_num: int, scenes: List[Scene]) -> bool:
        for sc in scenes:
            if sc.frame_start is not None and sc.frame_end is not None:
                if sc.frame_start <= frame_num <= sc.frame_end:
                    return True
        return False

    def _try_encode_ready_frames(self, writer: VideoWriter, scenes: List[Scene], pbar: Optional[tqdm] = None):
        while True:
            f = self.next_frame_to_encode
            if f not in self.frames_buffer:
                break  # need more frames read
            # Must have detection flag decided
            if f not in self.mosaic_detected_flags:
                break
            mosaic_flag = self.mosaic_detected_flags[f]
            if mosaic_flag:
                # need patches & no active scene covering it
                if self._frame_waiting_on_active_scene(f, scenes):
                    break
                if f not in self.frame_patches:
                    break
                # apply patches - work on a copy to avoid in-place modification issues
                frame = self.frames_buffer[f].clone()  # Create a copy to avoid modifying original
                for patch in self.frame_patches[f]:
                    t, l, b, r = patch.box
                    region = frame[t:b+1, l:r+1, :]
                    
                    # Ensure consistent data types for blending
                    # Convert region and patch.img to float32 for blending, then back to uint8
                    region_float = region.float()  # uint8 -> float32 (0-255)
                    patch_img_float = patch.img.float()  # uint8 -> float32 (0-255)
                    patch_mask = patch.mask  # already float32 (0-1)
                    
                    # Perform alpha blending in float32 for precision
                    blended = (region_float * (1.0 - patch_mask[..., None]) + patch_img_float * patch_mask[..., None])
                    blended = blended.clamp(0, 255).to(dtype=torch.uint8)
                    
                    frame[t:b+1, l:r+1, :] = blended
                del self.frame_patches[f]
            else:
                frame = self.frames_buffer[f]
            # Encode
            pts = self.frame_pts.get(f)
            writer.write(frame, frame_pts=pts)
            del self.frames_buffer[f]
            del self.mosaic_detected_flags[f]
            if f in self.frame_pts:
                del self.frame_pts[f]
            self.next_frame_to_encode += 1
            if pbar:
                pbar.update(1)

    def process(self, output_path: str):
        meta = self.video_metadata
        scenes: List[Scene] = []
        clip_counter = 0
        total_frames = meta.frames_count if meta.frames_count else None
        pbar = tqdm(total=total_frames, desc="Processing frames (streaming)")

        with VideoReader(self.video_file, self.batch_size, cuda_ctx, model_stream) as video_reader, VideoWriter(output_path, meta.video_width, meta.video_height, meta.video_fps_exact, cuda_ctx=cuda_ctx, modelstream=model_stream) as writer, torch.cuda.stream(model_stream):
            frames_iter = video_reader.frames()
            current_frame_num = 0
            eof = False
            while not eof:
                batch_orig_frames = []
                batch_pts = []
                for _ in range(self.batch_size):
                    try:
                        frame, pts = next(frames_iter)
                        batch_orig_frames.append(frame)
                        batch_pts.append(pts)
                    except StopIteration:
                        eof = True
                        break
                if len(batch_orig_frames) == 0 and eof:
                    break
                batch_pre = self.detection_model.preprocess(batch_orig_frames)
                preds_raw = self.detection_model.inference(batch_pre)
                batch_results = self.detection_model.postprocess(preds_raw, batch_pre, batch_orig_frames)

                for i, results in enumerate(batch_results):
                    frame_num = current_frame_num + i
                    self.frames_buffer[frame_num] = batch_orig_frames[i]
                    # store pts if available
                    if i < len(batch_pts):
                        self.frame_pts[frame_num] = batch_pts[i]
                    self._update_scenes_with_results(results, scenes, frame_num)
                    self._flush_completed_scenes(scenes, frame_num, eof=False, clip_counter_ref=[clip_counter])
                    # Attempt to encode frames now possible
                    self._try_encode_ready_frames(writer, scenes, pbar)
                current_frame_num += len(batch_results)
            # EOF flush
            self._flush_completed_scenes(scenes, current_frame_num, eof=True, clip_counter_ref=[clip_counter])
            # Encode any remaining frames
            self._try_encode_ready_frames(writer, scenes, pbar)
            # At EOF: any frames without mosaics but not yet encoded (shouldn't happen) -> encode
            remaining_frames = sorted(self.frames_buffer.keys())
            for f in remaining_frames:
                if f < self.next_frame_to_encode:
                    continue
                frame = self.frames_buffer[f].clone()  # Create a copy to avoid modifying original
                if f in self.frame_patches:
                    for patch in self.frame_patches[f]:
                        t, l, b, r = patch.box
                        region = frame[t:b+1, l:r+1, :]
                        
                        # Ensure consistent data types for blending
                        # Convert region and patch.img to float32 for blending, then back to uint8
                        region_float = region.float()  # uint8 -> float32 (0-255)
                        patch_img_float = patch.img.float()  # uint8 -> float32 (0-255)
                        patch_mask = patch.mask  # already float32 (0-1)
                        
                        # Perform alpha blending in float32 for precision
                        blended = (region_float * (1.0 - patch_mask[..., None]) + patch_img_float * patch_mask[..., None])
                        blended = blended.clamp(0, 255).to(dtype=torch.uint8)
                        
                        frame[t:b+1, l:r+1, :] = blended
                pts = self.frame_pts.get(f)
                writer.write(frame, frame_pts=pts)
                self.next_frame_to_encode = f + 1
                if f in self.frame_pts:
                    del self.frame_pts[f]
                if pbar:
                    pbar.update(1)
            # ensure progress reflects all frames
            # (pbar already updated incrementally)
            encoded_frames = self.next_frame_to_encode
            if total_frames and encoded_frames != total_frames:
                logger.info("Streaming pass encoded %d/%d frames (%.2f%%)", encoded_frames, total_frames, (encoded_frames/total_frames)*100)
        pbar.close()
        return output_path


def process_video_sequential_streaming(
    input_path: str,
    output_path: str,
    device: str,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: Optional[str],
    mosaic_detection_model_path: str,
    max_clip_length: int = 180,
    batch_size: int = 4,
    detection_only: bool = False
):
    mosaic_detection_model, mosaic_restoration_model, pad_mode = load_models(
        device,
        mosaic_restoration_model_name,
        mosaic_restoration_model_path,
        mosaic_restoration_config_path,
        mosaic_detection_model_path,
    )
    restorer = StreamingSequentialRestorer(
        video_file=input_path,
        detection_model=mosaic_detection_model,
        restoration_model=mosaic_restoration_model,
        restoration_model_name=mosaic_restoration_model_name,
        device=device,
        max_clip_length=max_clip_length,
        clip_size=256,
        pad_mode=pad_mode,
        batch_size=batch_size,
        detection_only=detection_only,
    )
    logger.info("Starting streaming sequential processing")
    return restorer.process(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- User adjustable block -------------------------------------------------
    INPUT_VIDEO = r"./test4-decensor-original.mp4"
    DEVICE = "cuda:0"
    MAX_CLIP_LENGTH = 240
    MOSAIC_RESTORATION_MODEL_NAME = "basicvsrpp"  # or "deepmosaics" variant
    MOSAIC_RESTORATION_MODEL_PATH = "model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"
    MOSAIC_RESTORATION_CONFIG_PATH = None  # use default
    # "v2 detection model" assumption: pick an existing v2 weight; adjust if actual filename differs
    MOSAIC_DETECTION_MODEL_PATH = "model_weights/lada_mosaic_detection_model_v2.pt"
    OUTPUT_VIDEO = INPUT_VIDEO.rsplit('.', 1)[0] + ".restored.streaming.mp4"
    BATCH_SIZE = 8
    # ---------------------------------------------------------------------------

    if not os.path.exists(MOSAIC_DETECTION_MODEL_PATH):
        logger.warning("Detection model path does not exist: %s", MOSAIC_DETECTION_MODEL_PATH)
    if not os.path.exists(MOSAIC_RESTORATION_MODEL_PATH):
        logger.warning("Restoration model path does not exist: %s", MOSAIC_RESTORATION_MODEL_PATH)

    process_video_sequential_streaming(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        device=DEVICE,
        mosaic_restoration_model_name=MOSAIC_RESTORATION_MODEL_NAME,
        mosaic_restoration_model_path=MOSAIC_RESTORATION_MODEL_PATH,
        mosaic_restoration_config_path=MOSAIC_RESTORATION_CONFIG_PATH,
        mosaic_detection_model_path=MOSAIC_DETECTION_MODEL_PATH,
        max_clip_length=MAX_CLIP_LENGTH,
        batch_size=BATCH_SIZE,
        detection_only=False
    )
    logger.info("Done. Output: %s", OUTPUT_VIDEO)
