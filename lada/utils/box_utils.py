import random

from lada.utils import Box, Image

def box_overlap(box1: Box, box2: Box):
    y1min, x1min, y1max, x1max = box1
    y2min, x2min, y2max, x2max = box2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max

def scale_box(img: Image, box: Box, mask_scale=1.0) -> Box:
    img_h, img_w = img.shape[:2]
    s = mask_scale - 1.0
    t, l, b, r = box
    w, h = r - l + 1, b - t + 1
    t -= h * s
    b += h * s
    l -= w * s
    r += w * s
    t = max(0, t)
    b = min(img_h - 1, b)
    l = max(0, l)
    r = min(img_w - 1, r)
    return int(t), int(l), int(b), int(r)

def random_scale_box(img: Image, box: Box, scale_range=(1.0, 1.5)) -> Box:
    scale = random.uniform(scale_range[0], scale_range[1])
    return scale_box(img, box, scale)

def convert_from_opencv(opencv_box: tuple[int, int, int, int]) -> Box:
    x, y, w, h = opencv_box
    t, l, b, r = y, x, y + h, x + w
    return t, l, b, r