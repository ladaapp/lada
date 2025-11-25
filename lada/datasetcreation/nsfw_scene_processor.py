# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import pathlib
import concurrent.futures as concurrent_futures
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

from lada.models.basicvsrpp.mosaic_video_dataset import create_degradation_pipeline
from lada.utils import mask_utils, Pad, Mask, Image
from lada.datasetcreation import restoration_dataset_metadata
from lada.utils import video_utils, image_utils
from lada.models.dover.evaluate import VideoQualityEvaluator
from lada.utils.image_utils import pad_image
from lada.datasetcreation.detectors.mosaic_classifier import MosaicClassifier
from lada.utils.mosaic_utils import get_random_parameter, addmosaic_base, get_mosaic_block_size_v1, \
    get_mosaic_block_size_v2, get_mosaic_block_size_v3
from lada.datasetcreation.nsfw_scene_detector import Scene, CroppedScene
from lada.datasetcreation.detectors.nudenet_nsfw_detector import NudeNetNsfwDetector
from lada.utils.threading_utils import wait_until_completed
from lada.datasetcreation.detectors.watermark_detector import WatermarkDetector


@dataclass
class SceneProcessingOptions:
    @dataclass
    class VideoQualityProcessingOptions:
        filter: bool
        add_metadata: bool
        min_quality: float

    @dataclass
    class WatermarkDetectionProcessingOptions:
        filter: bool
        add_metadata: bool

    @dataclass
    class NudeNetNsfwDetectionProcessingOptions:
        filter: bool
        add_metadata: bool

    @dataclass
    class CensorDetectionProcessingOptions:
        filter: bool
        add_metadata: bool

    output_dir: Path
    save_flat: bool
    out_size: int
    save_cropped: bool
    save_uncropped: bool
    resize_crops: bool
    preserve_crops: bool
    save_mosaic: bool
    degrade_mosaic: bool
    save_as_images: bool
    quality_evaluation: VideoQualityProcessingOptions
    watermark_detection: WatermarkDetectionProcessingOptions
    nudenet_nsfw_detection: NudeNetNsfwDetectionProcessingOptions
    censor_detection: CensorDetectionProcessingOptions

@dataclass
class SceneCreationOptions:
    stride_length: int
    scene_min_frames: str
    scene_max_frames: int
    random_extend_masks: bool

class MosaicRandomParams:
    def __init__(self, scene: Scene):
        box_sizes = [(r-l+1) * (b-t+1)for t,l,b,r in scene.get_boxes()]
        median_idx = np.argsort(box_sizes)[len(box_sizes) // 2]
        _, mask_image_representative, box = scene[median_idx]
        t, l, b, r = box
        filter = np.ones_like(mask_image_representative, dtype=bool)
        filter[t:b + 1, l:r + 1,:] = False
        if sum(mask_image_representative[filter]) > 0:
            print(f"non zero pixels outside of box: {sum(mask_image_representative[filter])}")
        mask_image_representative[filter] = 0
        self.mosaic_size, self.mosaic_mod, self.mosaic_rectangle_ratio, self.mosaic_feather_size = get_random_parameter(mask_image_representative)
        self.mosaic_mask_dilation_iterations = np.random.choice(range(2))

class DatasetItem:
    def __init__(self, cropped_scene: CroppedScene, scene: Optional[Scene], mosaic: bool, crop: bool, resize_crops: Optional[bool], resize_crop_size: Optional[int], mosaic_params: Optional[MosaicRandomParams], degrade: Callable, scene_type: Literal['cropped_scaled', 'cropped_unscaled', 'uncropped']):
        self._images: list[Image] = []
        self._masks: list[Mask] = []
        self._pads: list[Pad] = []
        self._meta: Optional[restoration_dataset_metadata.RestorationDatasetMetadataV2] = None
        self._quality_score: Optional[restoration_dataset_metadata.VisualQualityScoreV1] = None
        self._watermark_detected: Optional[bool] = None
        self._nudenet_nsfw_detected: Optional[bool] = None
        self._nudenet_nsfw_detected_classes: Optional[restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1] = None
        self._censoring_detected: Optional[bool] = None
        self._scene_type: Literal['cropped_scaled', 'cropped_unscaled', 'uncropped'] = scene_type

        self._init_images_masks_and_pad(cropped_scene, scene, mosaic, crop, resize_crops, resize_crop_size, mosaic_params, degrade)

    @property
    def images(self):
        return self._images

    @property
    def masks(self):
        return self._masks

    @property
    def pads(self):
        return self._pads

    @property
    def quality_score(self):
        return self._quality_score

    @quality_score.setter
    def quality_score(self, value):
        self._quality_score = value

    @property
    def watermark_detected(self):
        return self._watermark_detected

    @watermark_detected.setter
    def watermark_detected(self, value):
        self._watermark_detected = value

    @property
    def nudenet_nsfw_detected(self):
        return self._nudenet_nsfw_detected

    @nudenet_nsfw_detected.setter
    def nudenet_nsfw_detected(self, value):
        self._nudenet_nsfw_detected = value

    @property
    def nudenet_nsfw_detected_classes(self):
        return self._nudenet_nsfw_detected_classes

    @nudenet_nsfw_detected_classes.setter
    def nudenet_nsfw_detected_classes(self, value):
        self._nudenet_nsfw_detected_classes = value

    @property
    def censoring_detected(self):
        return self._censoring_detected

    @censoring_detected.setter
    def censoring_detected(self, value):
        self._censoring_detected = value

    def _init_images_masks_and_pad(self, cropped_scene: CroppedScene, scene: Optional[Scene], mosaic: bool, crop: bool, resize_crops: bool, resize_crop_size: Optional[int], mosaic_params: Optional[MosaicRandomParams], degrade: Callable):

        for i in range(len(cropped_scene)):
            if mosaic:
                scene_image, scene_mask, _ = cropped_scene[i]
                mask = mask_utils.dilate_mask(scene_mask, iterations=mosaic_params.mosaic_mask_dilation_iterations)
                cropped_image, cropped_mask = addmosaic_base(scene_image, mask, mosaic_params.mosaic_size,
                                                                           model=mosaic_params.mosaic_mod, rect_ratio=mosaic_params.mosaic_rectangle_ratio,
                                                                           feather=mosaic_params.mosaic_feather_size)
            else:
                cropped_image, cropped_mask, _ = cropped_scene[i]

            if crop:
                image, mask = cropped_image, cropped_mask
                if mosaic and degrade:
                    # todo: running mosaic restoration dataset degradation pipeline on single images will produce different results compared to passing intended list of consecutive frames to it
                    #  either refactor this code or just remove it. its only used for debugging, degraded frames should be generated dynamically in the dataset class instead of pre-generated via dataset creation script.
                    shape_before_degrade = image.shape
                    image = degrade(image)
                    image = cv2.resize(np.array(image), (shape_before_degrade[1], shape_before_degrade[0]))
                if resize_crops:
                    mask = image_utils.resize(mask, resize_crop_size, interpolation=cv2.INTER_NEAREST)
                    image = image_utils.resize(image, resize_crop_size, interpolation=cv2.INTER_CUBIC)
                    mask, _ = pad_image(mask, resize_crop_size, resize_crop_size, mode='zero')
                    image, pad = pad_image(image, resize_crop_size, resize_crop_size, mode='zero')
                else:
                    max_width, max_height = cropped_scene.get_max_width_height()
                    mask, _ = pad_image(mask, max_height, max_width, mode='zero')
                    image, pad = pad_image(image, max_height, max_width, mode='zero')
            else:
                scene_image, scene_mask, scene_box = scene[i]
                if mosaic:
                    _, _, scene_box = cropped_scene[i]
                    t, l, b, r = scene_box
                    image = scene_image.copy()
                    image[t:b + 1, l:r + 1, :] = cropped_image
                    if degrade:
                        # todo: degradation pipeline currently is optimized for 256x256 video crops not arbitrary large vids. lets skip degradations for now
                        pass
                    mask = np.zeros_like(scene_mask, dtype=scene_mask.dtype)
                    mask[t:b + 1, l:r + 1] = cropped_mask
                else:
                    image, mask = scene_image, scene_mask
                pad = [0, 0, 0, 0]

            self._images.append(image)
            self._masks.append(mask)
            self._pads.append(pad)

        assert len(cropped_scene) == len(self._images) == len(self._masks) == len(self._pads), f"number of images, masks and pads are not the same: {len(cropped_scene)} == {len(self._images)} == {len(self._masks)} == {len(self._pads)}"

    def _save_vid(self, file_path: pathlib.Path, imgs: list[np.ndarray[np.uint8]], fps=30, gray=False):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(file_path.absolute())
        if gray:
            video_utils.write_masks_to_video_file(imgs, path, fps)
        else:
            video_utils.write_frames_to_video_file(imgs, path, fps)

    def _save_meta(self, file_path: pathlib.Path, meta: restoration_dataset_metadata.RestorationDatasetMetadataV2):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(file_path.absolute())
        meta.to_json_file(path)

    def _save_imgs(self, file_path_template_format_string: pathlib.Path, imgs: np.ndarray, jpeg_quality_level=95):
        file_path_template_format_string.parent.mkdir(parents=True, exist_ok=True)
        file_ext = file_path_template_format_string
        for i in range(len(imgs)):
            path = str(Path(str(file_path_template_format_string).format(i)).absolute())
            try:
                if file_ext == ".jpg":
                    cv2.imwrite(path, imgs[i], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_level])
                else:
                    cv2.imwrite(path, imgs[i])
                    cv2.imwrite(path, imgs[i])
            except Exception as e:
                print(e)

    def _get_base_mosaic_block_size(self, scene: Scene) -> restoration_dataset_metadata.MosaicBlockSizeV2:
        box_sizes = [(r - l + 1) * (b - t + 1) for t, l, b, r in scene.get_boxes()]
        median_idx = np.argsort(box_sizes)[len(box_sizes) // 2]
        _, mask_image_representative, box = scene[median_idx]
        t, l, b, r = box
        filter = np.ones_like(mask_image_representative, dtype=bool)
        filter[t:b + 1, l:r + 1, :] = False
        mask_image_representative[filter] = 0

        # not sure what we'll use later, lets save all variants for now

        mosaic_block_size = restoration_dataset_metadata.MosaicBlockSizeV2(
            mosaic_size_v3=get_mosaic_block_size_v3(
                (scene.video_meta_data.video_height, scene.video_meta_data.video_width)),
            mosaic_size_v2=get_mosaic_block_size_v2(mask_image_representative),
            mosaic_size_v1_normal=get_mosaic_block_size_v1(mask_image_representative, 'normal'),
            mosaic_size_v1_bounding=get_mosaic_block_size_v1(mask_image_representative, 'bounding'))
        return mosaic_block_size

    def _get_io_path(self, output_dir:pathlib.Path, scene_type: Literal['cropped_scaled', 'cropped_unscaled', 'uncropped'], scene:Scene, save_as_images:bool, save_flat:bool, file_type: Literal['mask', 'img', 'meta'], mosaic: bool) -> pathlib.Path:
        file_suffix = '-'
        subdir_name = 'crop_scaled' if scene_type == 'cropped_scaled' else 'crop_unscaled' if scene_type == 'cropped_unscaled' else 'orig'
        if not (mosaic and file_type == 'img'):
            subdir_name += ('_' + file_type)
        if mosaic:
            subdir_name += "_mosaic"
        file_name = pathlib.Path(scene.video_meta_data.video_file).name
        if save_flat:
            file_prefix = f"{file_name}-{scene.id:06d}{file_suffix}"
            frame_dir = output_dir.joinpath(subdir_name)
        else:
            if save_as_images:
                file_prefix = ""
                frame_dir = output_dir.joinpath(subdir_name).joinpath(file_name).joinpath(f"{scene.id:06d}")
            else:
                file_prefix = f"{scene.id:06d}"
                frame_dir = output_dir.joinpath(subdir_name).joinpath(file_name)
        if file_type == 'img':
            if save_as_images:
                file_extension = ".jpg"
                file_name_format_string = file_prefix + '{%06d}' + file_extension
                file_name = file_name_format_string
            else:
                file_extension = ".mp4"
                file_name = file_prefix + file_extension
        elif file_type == 'mask':
            if save_as_images:
                file_extension = ".png"
                file_name_format_string = file_prefix + '{%06d}' + file_extension
                file_name = file_name_format_string
            else:
                file_extension = ".mkv"
                file_name = file_prefix + file_extension
        elif file_type == 'meta':
            file_extension = ".json"
            file_name = file_prefix + file_extension
        else:
            raise TypeError("expected file_type to be one of: 'mask', 'img' or 'meta'")
        return frame_dir.joinpath(file_name)

    def init_meta(self, scene: Scene, output_dir, save_as_images: bool, save_flat: bool, mosaic: bool, mosaic_params: MosaicRandomParams):
        def _get_relative_path_dir(scene_type, mosaic):
            metadata_path = self._get_io_path(output_dir, scene_type, scene, save_as_images, save_flat, 'meta', mosaic)
            img_path = self._get_io_path(output_dir, scene_type, scene, save_as_images, save_flat, 'img', mosaic)
            mask_path = self._get_io_path(output_dir, scene_type, scene, save_as_images, save_flat, 'mask', mosaic)
            return str(img_path.relative_to(metadata_path.parent, walk_up=True)), str(mask_path.relative_to(metadata_path.parent, walk_up=True))

        mosaic_metadata = restoration_dataset_metadata.MosaicMetadataV1(mod=mosaic_params.mosaic_mod,
                                                                        rect_ratio=mosaic_params.mosaic_rectangle_ratio,
                                                                        mosaic_size=mosaic_params.mosaic_size,
                                                                        feather_size=mosaic_params.mosaic_feather_size) if mosaic else None

        relative_nsfw_video_path, relative_mask_video_path = _get_relative_path_dir(self._scene_type, False)
        relative_mosaic_nsfw_video_path, relative_mosaic_mask_video_path = _get_relative_path_dir(self._scene_type,True) if mosaic else (None, None)
        self._meta = restoration_dataset_metadata.RestorationDatasetMetadataV2(
            name=pathlib.Path(scene.video_meta_data.video_file).name,
            fps=scene.video_meta_data.video_fps,
            frames_count=len(scene),
            orig_shape=(scene.video_meta_data.video_height, scene.video_meta_data.video_width),
            scene_shape=self._images[0].shape[:2],
            base_mosaic_block_size=self._get_base_mosaic_block_size(scene),
            pad=self._pads,
            relative_nsfw_video_path=relative_nsfw_video_path,
            relative_mask_video_path=relative_mask_video_path,
            relative_mosaic_nsfw_video_path=relative_mosaic_nsfw_video_path,
            relative_mosaic_mask_video_path=relative_mosaic_mask_video_path,
            mosaic=mosaic_metadata,
            video_quality=self._quality_score,
            watermark_detected=self._watermark_detected,
            nudenet_nsfw_detected=self._nudenet_nsfw_detected,
            nudenet_nsfw_detected_classes=self._nudenet_nsfw_detected_classes,
            censoring_detected=self._censoring_detected,
        )

    def save(self, output_dir, scene, mosaic, save_as_images, save_flat, target_fps, io_executor) -> list[concurrent_futures.Future]:
        io_futures = []

        def _save(data, file_type: Literal['mask', 'img', 'meta'], mosaic):
            file_path = self._get_io_path(output_dir, self._scene_type, scene, save_as_images, save_flat, file_type, mosaic)
            if file_type == 'meta':
                io_futures.append(io_executor.submit(self._save_meta, file_path, data))
            else:
                if save_as_images:
                    io_futures.append(io_executor.submit(self._save_imgs, file_path, data))
                else:
                    gray = file_type == 'mask'
                    io_futures.append(io_executor.submit(self._save_vid, file_path, data, target_fps, gray))
        if self._meta:
            _save(self._meta, 'meta', mosaic)
        _save(self._images, 'img', mosaic)
        _save(self._masks, 'mask', mosaic)
        return io_futures

class SceneProcessor:
    def __init__(self,  video_quality_evaluator: VideoQualityEvaluator, watermark_detector: WatermarkDetector, nudenet_nsfw_detector: NudeNetNsfwDetector, censoring_detector: MosaicClassifier):
        self.video_quality_evaluator = video_quality_evaluator
        self.watermark_detector = watermark_detector
        self.nudenet_nsfw_detector = nudenet_nsfw_detector
        self.censor_detector = censoring_detector

    def process_scene(self, scene: Scene, output_dir: Path, scene_processing_options: SceneProcessingOptions):
        print("Started processing scene", scene.id)
        cropped_scene = CroppedScene(scene, target_size=(scene_processing_options.out_size,scene_processing_options.out_size), border_size=0.08)

        dataset_item_mosaic_crop_unscaled: Optional[DatasetItem] = None
        dataset_item_mosaic_crop_scaled: Optional[DatasetItem] = None
        dataset_item_mosaic_uncropped: Optional[DatasetItem] = None
        dataset_item_crop_unscaled: Optional[DatasetItem] = None
        dataset_item_crop_scaled: Optional[DatasetItem] = None
        dataset_item_uncropped: Optional[DatasetItem] = None

        mosaic_params = MosaicRandomParams(scene) if scene_processing_options.save_mosaic else None

        #########
        ## Images, Masks, Pads
        #########
        if scene_processing_options.save_mosaic:
            mosaic_params = MosaicRandomParams(scene)
            degrade = create_degradation_pipeline(scene_processing_options.out_size)
            if scene_processing_options.save_cropped:
                if scene_processing_options.preserve_crops:
                    dataset_item_mosaic_crop_unscaled = DatasetItem(cropped_scene, scene, True, True, False, scene_processing_options.out_size, mosaic_params, degrade, 'cropped_unscaled')
                if scene_processing_options.resize_crops:
                    dataset_item_mosaic_crop_scaled = DatasetItem(cropped_scene, scene, True, True, True, scene_processing_options.out_size, mosaic_params, degrade, 'cropped_scaled')
            if scene_processing_options.save_uncropped:
                dataset_item_mosaic_uncropped = DatasetItem(cropped_scene, scene, True, False, None, None, mosaic_params, degrade, 'uncropped')
        if scene_processing_options.save_cropped:
            if scene_processing_options.save_cropped:
                if scene_processing_options.preserve_crops:
                    dataset_item_crop_unscaled = DatasetItem(cropped_scene, scene, False, True, False, scene_processing_options.out_size, None, None,'cropped_unscaled')
                if scene_processing_options.resize_crops:
                    dataset_item_crop_scaled = DatasetItem(cropped_scene, scene, False, True, True, scene_processing_options.out_size, None, None, 'cropped_scaled')
            if scene_processing_options.save_uncropped:
                dataset_item_uncropped = DatasetItem(cropped_scene, scene, False, False, None, None, None, None, 'uncropped')

        scene_analyzers_futures = []
        with concurrent_futures.ThreadPoolExecutor() as scene_analyzers_executor:
            #########
            ## Video quality evaluation
            #########
            def _run_video_quality_evaluation():
                if scene_processing_options.quality_evaluation.filter or scene_processing_options.quality_evaluation.add_metadata:
                    if scene_processing_options.save_cropped:
                        if scene_processing_options.resize_crops:
                            score = self.video_quality_evaluator.evaluate(dataset_item_crop_scaled.images)
                            dataset_item_crop_scaled.quality_score = restoration_dataset_metadata.VisualQualityScoreV1(**score)
                        if scene_processing_options.preserve_crops:
                            score = self.video_quality_evaluator.evaluate(dataset_item_crop_unscaled.images)
                            dataset_item_crop_unscaled.quality_score = restoration_dataset_metadata.VisualQualityScoreV1(**score)
                    if scene_processing_options.save_uncropped:
                        score = self.video_quality_evaluator.evaluate(dataset_item_uncropped.images)
                        dataset_item_uncropped.quality_score = restoration_dataset_metadata.VisualQualityScoreV1(**score)
            scene_analyzers_futures.append(scene_analyzers_executor.submit(_run_video_quality_evaluation))

            #########
            ## Watermark detection
            #########
            def _run_watermark_detection():
                if scene_processing_options.watermark_detection.filter or scene_processing_options.watermark_detection.add_metadata:
                    if scene_processing_options.save_cropped:
                        _watermark_detected = self.watermark_detector.detect(scene.get_images(), cropped_scene.get_boxes())
                        if scene_processing_options.resize_crops:
                            dataset_item_crop_scaled.watermark_detected = _watermark_detected
                        if scene_processing_options.preserve_crops:
                            dataset_item_crop_unscaled.watermark_detected = _watermark_detected
                    if scene_processing_options.save_uncropped:
                        dataset_item_uncropped.watermark_detected = self.watermark_detector.detect(scene.get_images())
            scene_analyzers_futures.append(scene_analyzers_executor.submit(_run_watermark_detection))

            #########
            ## NudeNet NSFW detection
            #########
            def _run_nudenet_nsfw_detection():
                if scene_processing_options.nudenet_nsfw_detection.filter or scene_processing_options.nudenet_nsfw_detection.add_metadata:
                    if scene_processing_options.save_cropped:
                        _nudenet_nsfw_detected, _nsfw_male_detected, _nsfw_female_detected = self.nudenet_nsfw_detector.detect(scene.get_images(), cropped_scene.get_boxes())
                        if scene_processing_options.resize_crops:
                            dataset_item_crop_scaled.nudenet_nsfw_detected, dataset_item_crop_scaled.nudenet_nsfw_detected_classes = _nudenet_nsfw_detected, restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1(_nsfw_male_detected, _nsfw_female_detected)
                        if scene_processing_options.preserve_crops:
                            dataset_item_crop_unscaled.nudenet_nsfw_detected, dataset_item_crop_unscaled.nudenet_nsfw_detected_classes = _nudenet_nsfw_detected, restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1(_nsfw_male_detected, _nsfw_female_detected)
                    if scene_processing_options.save_uncropped:
                        _nudenet_nsfw_detected, _nsfw_male_detected, _nsfw_female_detected = self.nudenet_nsfw_detector.detect(scene.get_images())
                        dataset_item_uncropped.nudenet_nsfw_detected, dataset_item_uncropped.nudenet_nsfw_detected_classes = _nudenet_nsfw_detected, restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1(_nsfw_male_detected, _nsfw_female_detected)
            scene_analyzers_futures.append(scene_analyzers_executor.submit(_run_nudenet_nsfw_detection))

            #########
            ## Censor detection
            #########
            def _run_censor_detection():
                if scene_processing_options.censor_detection.filter or scene_processing_options.censor_detection.add_metadata:
                    if scene_processing_options.save_cropped:
                        _censoring_detected = self.censor_detector.detect(scene.get_images(), cropped_scene.get_boxes())
                        if scene_processing_options.resize_crops:
                            dataset_item_crop_scaled.censoring_detected = _censoring_detected
                        if scene_processing_options.preserve_crops:
                            dataset_item_crop_unscaled.censoring_detected = _censoring_detected
                    if scene_processing_options.save_uncropped:
                        dataset_item_uncropped.censoring_detected = self.censor_detector.detect(scene.get_images())
            scene_analyzers_futures.append(scene_analyzers_executor.submit(_run_censor_detection))

        wait_until_completed(scene_analyzers_futures)

        #########
        ## META
        #########
        if scene_processing_options.save_cropped:
            if scene_processing_options.resize_crops:
                dataset_item_crop_scaled.init_meta(scene, output_dir, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene_processing_options.save_mosaic, mosaic_params)
            if scene_processing_options.preserve_crops:
                dataset_item_crop_unscaled.init_meta(scene, output_dir, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene_processing_options.save_mosaic, mosaic_params)
        if scene_processing_options.save_uncropped:
            dataset_item_uncropped.init_meta(scene, output_dir, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene_processing_options.save_mosaic, mosaic_params)

        #########
        ## Filtering
        #########
        if scene_processing_options.save_cropped and scene_processing_options.preserve_crops:
            _filtering_dataset_item = dataset_item_crop_unscaled
        elif scene_processing_options.save_cropped and scene_processing_options.resize_crops:
            _filtering_dataset_item = dataset_item_crop_scaled
        elif scene_processing_options.save_uncropped:
            _filtering_dataset_item = dataset_item_uncropped
        else:
            _filtering_dataset_item = None

        if scene_processing_options.quality_evaluation.filter and _filtering_dataset_item and _filtering_dataset_item.quality_score.overall < scene_processing_options.quality_evaluation.min_quality:
            print(f"Skipped scene {scene.id} because of low visual video quality ({_filtering_dataset_item.quality_score.overall:.4f} < {scene_processing_options.quality_evaluation.min_quality})")
            return
        elif scene_processing_options.watermark_detection.filter and _filtering_dataset_item and _filtering_dataset_item.watermark_detected:
            print(f"Skipped scene {scene.id} because watermark(s) have been detected")
            return
        elif scene_processing_options.nudenet_nsfw_detection.filter and _filtering_dataset_item and not _filtering_dataset_item.nudenet_nsfw_detected:
            print(f"Skipped scene {scene.id} because not NSFW according to NudeNetNsfwDetector")
            return
        elif scene_processing_options.censor_detection.filter and _filtering_dataset_item and _filtering_dataset_item.censoring_detected:
            print(f"Skipped scene {scene.id} because censoring has been detected")
            return

        #########
        ## Save
        #########
        io_futures = []
        with concurrent_futures.ThreadPoolExecutor() as io_executor:
            if scene_processing_options.save_mosaic:
                if scene_processing_options.save_cropped:
                    if scene_processing_options.preserve_crops:
                        io_futures.extend(dataset_item_mosaic_crop_unscaled.save(output_dir, scene, True, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene.video_meta_data.video_fps, io_executor))
                    if scene_processing_options.resize_crops:
                        io_futures.extend(dataset_item_mosaic_crop_scaled.save(output_dir, scene, True, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor))
                if scene_processing_options.save_uncropped:
                    io_futures.extend(dataset_item_mosaic_uncropped.save(output_dir, scene, True, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor))
            if scene_processing_options.save_cropped:
                if scene_processing_options.preserve_crops:
                    io_futures.extend(dataset_item_crop_unscaled.save(output_dir, scene, False, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor))
                if scene_processing_options.resize_crops:
                    io_futures.extend(dataset_item_crop_scaled.save(output_dir, scene, False, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor))
            if scene_processing_options.save_uncropped:
                io_futures.extend(dataset_item_uncropped.save(output_dir, scene, False, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor))
        wait_until_completed(io_futures)
        print("Finished processing scene", scene.id)