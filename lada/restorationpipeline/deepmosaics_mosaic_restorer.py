import torch

from lada.models.deepmosaics.inference import restore_video_frames

class DeepmosaicsMosaicRestorer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.dtype = model.dtype

    def restore(self, video: list[torch.Tensor]) -> list[torch.Tensor]:
        frames = [x.contiguous().numpy() for x in video]
        restored_frames = restore_video_frames(self.device.index, self.model, frames)
        return [torch.from_numpy(x) for x in restored_frames]

