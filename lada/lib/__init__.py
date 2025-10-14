from dataclasses import dataclass
from fractions import Fraction

import numpy as np

"""
A bounding box of a detected object defined by two points, the top/left and bottom/right pixel.
Represented as X/Y coordinate tuple: top-left (Y), top-left (X), bottom-right (Y), bottom-right (X)
"""
type Box = tuple[int, int, int, int]

"""
A segmentation mask of a detected object. Pixel values of 0 indicate that the pixel is not part of the object.
Shape: (H, W, 1)
"""
type Mask = np.ndarray[np.uint8]

"""
Color Image
Shape: (H, W, C)
H, W, C stand for image height, width and color channels respectively. C is always 3 and must be in BGR instead of RGB order 
"""
type Image = np.ndarray[np.uint8]

"""
Padding of an Image or Mask represented as tuple padding values (number of black pixels) added to each image edge:
(padding-top, padding-bottom, padding-left, padding-right)
"""
type Pad = tuple[int, int, int, int]

"""
Metadata about a video file
"""
@dataclass
class VideoMetadata:
    video_file: str
    video_height: int
    video_width: int
    video_fps: float
    average_fps: float
    video_fps_exact: Fraction
    codec_name: str
    frames_count: int
    duration: float
    time_base: Fraction
    start_pts: int
    is_vfr: bool