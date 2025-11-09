from lada.lib import Box

def box_overlap(box1: Box, box2: Box):
    y1min, x1min, y1max, x1max = box1
    y2min, x2min, y2max, x2max = box2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max