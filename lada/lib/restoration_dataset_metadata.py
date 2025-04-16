import os
from abc import ABC
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional
from lada.lib import Pad, mosaic_utils

@dataclass
class MosaicMetadataV1:
    mod: str
    rect_ratio: float
    mosaic_size: int
    feather_size: float

@dataclass
class MosaicBlockSizeV1:
    mosaic_size_v2: float
    mosaic_size_v1_normal: float
    mosaic_size_v1_bounding: float

@dataclass
class MosaicBlockSizeV2:
    mosaic_size_v3: float
    mosaic_size_v2: float
    mosaic_size_v1_normal: float
    mosaic_size_v1_bounding: float

@dataclass
class VisualQualityScoreV1:
    aesthetic: float
    technical: float
    overall: float

@dataclass
class NudeNetNsfwClassDetectionsV1:
    MALE_GENITALIA_EXPOSED: bool
    FEMALE_GENITALIA_EXPOSED: bool

@dataclass
class AbstractRestorationDatasetMetadata(ABC):
    def __init__(self):
        self.version = None

    def read_metadata_version(path: str) -> int:
        with open(path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
        return json_dict['version'] if 'version' in json_dict else 1

    def to_json_file(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json_dict = asdict(self)
            json_dict["version"] = self.version
            json.dump(json_dict, f)

    def from_json_file(path: str):
        raise NotImplementedError()

@dataclass
class RestorationDatasetMetadataV1(AbstractRestorationDatasetMetadata):
    version = 1
    fps: int
    frames_count: Optional[int]
    name: str
    orig_width: int
    orig_height: int
    base_mosaic_block_size: Optional[MosaicBlockSizeV1]
    mosaic: Optional[MosaicMetadataV1]
    pad: Optional[list[Pad]]
    height: int
    width: int
    video_quality: VisualQualityScoreV1
    frame_count: Optional[int] = None # deprecated

    def from_json_file(path: str):
        with open(path, 'r') as f:
            json_dict = json.load(f)
        version = json_dict['version'] if json_dict.get('version') else 1
        assert version == 1, "Cannot read metadata version " + version
        return RestorationDatasetMetadataV1(
            json_dict["fps"],
            json_dict.get("frames_count"),
            json_dict["name"],
            json_dict["orig_width"],
            json_dict["orig_height"],
            MosaicBlockSizeV1(
                mosaic_size_v2=json_dict["base_mosaic_block_size"]["mosaic_size_v2"],
                mosaic_size_v1_normal=json_dict["base_mosaic_block_size"]["mosaic_size_v1_normal"],
                mosaic_size_v1_bounding=json_dict["base_mosaic_block_size"]["mosaic_size_v1_bounding"],
            ) if json_dict.get("base_mosaic_block_size") else None,
            MosaicMetadataV1(
                mod=json_dict["mosaic"]["mod"],
                rect_ratio=json_dict["mosaic"]["rect_ratio"],
                mosaic_size=json_dict["mosaic"]["mosaic_size"],
                feather_size=json_dict["mosaic"]["feather_size"],
            ) if json_dict.get('mosaic') else None,
            json_dict.get("pad"),
            json_dict["height"],
            json_dict["width"],
            VisualQualityScoreV1(
                json_dict["video_quality"]["aesthetic"],
                json_dict["video_quality"]["technical"],
                json_dict["video_quality"]["overall"],
            ),
            json_dict.get("frame_count"),
        )

@dataclass
class RestorationDatasetMetadataV2(AbstractRestorationDatasetMetadata):
    version = 2
    name: str
    fps: float | int
    frames_count: int
    orig_shape: tuple[int, int]
    scene_shape: tuple[int, int]
    base_mosaic_block_size: MosaicBlockSizeV2
    pad: list[Pad]
    relative_nsfw_video_path: str
    relative_mask_video_path: str
    relative_mosaic_nsfw_video_path: Optional[str]
    relative_mosaic_mask_video_path: Optional[str]
    mosaic: Optional[MosaicMetadataV1]
    video_quality: Optional[VisualQualityScoreV1]
    watermark_detected: Optional[bool]
    nudenet_nsfw_detected: Optional[bool]
    nudenet_nsfw_detected_classes: Optional[NudeNetNsfwClassDetectionsV1]
    censoring_detected: Optional[bool]

    def _determine_relative_file_paths_by_v1_metadata(path: str, v1_metadata: RestorationDatasetMetadataV1) -> tuple[str, str, Optional[str], Optional[str]]:
        metadata_pathlib_path = Path(path)

        base_dir: Path = metadata_pathlib_path.parent.parent
        metadata_dir_name: str = metadata_pathlib_path.parent.name
        filename_stem: str = f"{os.path.splitext(os.path.basename(path))[0]}"

        video_dir_path = base_dir / metadata_dir_name.replace('_meta', '_img')
        video_pathlib_path = video_dir_path / (filename_stem + ".mp4")
        assert video_pathlib_path.exists(), "cannot convert v1 metadata to v2: couldn't find video file"

        mask_video_dir_path = base_dir / metadata_dir_name.replace('_meta', '_mask')
        mask_video_pathlib_path = mask_video_dir_path / (filename_stem + ".mkv")
        assert mask_video_pathlib_path.exists(), "cannot convert v1 metadata to v2: couldn't find mask video file"

        if v1_metadata.mosaic:
            mosaic_video_dir_path = base_dir / metadata_dir_name.replace('_meta',
                                                                                                             '_mosaic')
            mosaic_video_pathlib_path = mosaic_video_dir_path / (filename_stem + ".mp4")
            assert mosaic_video_pathlib_path.exists(), "cannot convert v1 metadata to v2: couldn't find mosaic video file"

            mask_mosaic_video_dir_path = base_dir / metadata_dir_name.replace('_meta', '_mask_mosaic')
            mask_mosaic_video_pathlib_path = mask_mosaic_video_dir_path / (filename_stem + ".mkv")
            assert mask_mosaic_video_pathlib_path.exists(), "cannot convert v1 metadata to v2: couldn't find mosaic video file"

            return (str(video_pathlib_path.relative_to(metadata_pathlib_path.parent, walk_up=True)),
                    str(mask_video_pathlib_path.relative_to(metadata_pathlib_path.parent, walk_up=True)),
                    str(mosaic_video_pathlib_path.relative_to(metadata_pathlib_path.parent, walk_up=True)),
                    str(mask_mosaic_video_pathlib_path.relative_to(metadata_pathlib_path.parent, walk_up=True)))
        else:
            return (str(video_pathlib_path.relative_to(metadata_pathlib_path.parent, walk_up=True)),
                    str(mask_video_pathlib_path.relative_to(metadata_pathlib_path.parent, walk_up=True)),
                    None,
                    None)

    def from_json_file(path: str):
        with open(path, 'r') as f:
            json_dict = json.load(f)
        version = json_dict['version'] if json_dict.get('version') else 1
        assert version in (1,2), "Cannot read metadata version " + version

        if version == 1:
            v1_metadata = RestorationDatasetMetadataV1.from_json_file(path)
            assert v1_metadata.pad, "cannot convert v1 metadata to v2: padding info missing"

            relative_nsfw_video_path, relative_mask_video_path, relative_mosaic_nsfw_video_path, relative_mosaic_mask_video_path = RestorationDatasetMetadataV2._determine_relative_file_paths_by_v1_metadata(path, v1_metadata)

            return RestorationDatasetMetadataV2(
                v1_metadata.name,
                v1_metadata.fps,
                v1_metadata.frames_count if v1_metadata.frames_count else v1_metadata.frame_count,
                (v1_metadata.orig_height, v1_metadata.orig_width),
                (v1_metadata.height, v1_metadata.width),
                MosaicBlockSizeV2(
                    mosaic_size_v3=mosaic_utils.get_mosaic_block_size_v3((v1_metadata.orig_height, v1_metadata.orig_width)),
                    mosaic_size_v2=v1_metadata.base_mosaic_block_size.mosaic_size_v2,
                    mosaic_size_v1_normal=v1_metadata.base_mosaic_block_size.mosaic_size_v1_normal,
                    mosaic_size_v1_bounding=v1_metadata.base_mosaic_block_size.mosaic_size_v1_bounding,
                ),
                v1_metadata.pad,
                relative_nsfw_video_path,
                relative_mask_video_path,
                relative_mosaic_nsfw_video_path,
                relative_mosaic_mask_video_path,
                v1_metadata.mosaic,
                v1_metadata.video_quality,
                None,
                None,
                None,
                None,
            )
        elif version == 2:
            return RestorationDatasetMetadataV2(
                json_dict["name"],
                json_dict["fps"],
                json_dict["frames_count"],
                json_dict["orig_shape"],
                json_dict["scene_shape"],
                MosaicBlockSizeV2(
                    mosaic_size_v3=json_dict["base_mosaic_block_size"]["mosaic_size_v3"],
                    mosaic_size_v2=json_dict["base_mosaic_block_size"]["mosaic_size_v2"],
                    mosaic_size_v1_normal=json_dict["base_mosaic_block_size"]["mosaic_size_v1_normal"],
                    mosaic_size_v1_bounding=json_dict["base_mosaic_block_size"]["mosaic_size_v1_bounding"],
                ),
                json_dict["pad"],
                json_dict["relative_nsfw_video_path"],
                json_dict["relative_mask_video_path"],
                json_dict.get("relative_mosaic_nsfw_video_path"),
                json_dict.get("relative_mosaic_mask_video_path"),
                MosaicMetadataV1(
                    mod=json_dict["mosaic"]["mod"],
                    rect_ratio=json_dict["mosaic"]["rect_ratio"],
                    mosaic_size=json_dict["mosaic"]["mosaic_size"],
                    feather_size=json_dict["mosaic"]["feather_size"],
                ) if json_dict.get("mosaic") else None,
                VisualQualityScoreV1(
                    json_dict["video_quality"]["aesthetic"],
                    json_dict["video_quality"]["technical"],
                    json_dict["video_quality"]["overall"],
                ) if json_dict.get("video_quality") else None,
                json_dict.get("watermark_detected"),
                json_dict.get("nudenet_nsfw_detected"),
                NudeNetNsfwClassDetectionsV1(
                    MALE_GENITALIA_EXPOSED=json_dict["nudenet_nsfw_detected_classes"]["MALE_GENITALIA_EXPOSED"],
                    FEMALE_GENITALIA_EXPOSED=json_dict["nudenet_nsfw_detected_classes"]["FEMALE_GENITALIA_EXPOSED"],
                ) if json_dict.get("nudenet_nsfw_detected_classes") else None,
                json_dict.get("censoring_detected"),
            )