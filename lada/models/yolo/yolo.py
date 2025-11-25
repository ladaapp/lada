from ultralytics.models import YOLO

class Yolo(YOLO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)