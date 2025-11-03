DATA = dict(
    nc=2,  # number of classes (two class: human body, human head)
    num_offsets=2,  # number of coordinates introduced by the body part, e.g., (head_x, head_y)
    names=[ 'person', 'head' ],  # class names.
    conf_thres_part=0.45, # the larger conf threshold for filtering body-part detection proposals
    iou_thres_part=0.75, # the smaller iou threshold for filtering body-part detection proposals
    match_iou_thres=0.6, # whether a body-part in matched with one body bbox
)