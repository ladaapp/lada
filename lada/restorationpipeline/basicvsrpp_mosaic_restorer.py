import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

    def restore(self, video: list[ImageTensor], max_frames=-1) -> list[ImageTensor]:
        return self.restore_batch([video], max_frames=max_frames)[0]

    def restore_batch(self, videos: list[list[ImageTensor]], max_frames=-1) -> list[list[ImageTensor]]:
        assert len(videos) > 0
        input_frame_count = len(videos[0])
        input_frame_shape = videos[0][0].shape
        for video in videos:
            assert len(video) == input_frame_count
            assert video[0].shape == input_frame_shape
        with torch.inference_mode():
            result = []
            inference_view = torch.stack(
                [
                    torch.stack([frame.permute(2, 0, 1) for frame in video], dim=0)
                    for video in videos
                ],
                dim=0,
            ).to(device=self.device).to(dtype=self.dtype).div_(255.0)

            if max_frames > 0:
                for i in range(0, inference_view.shape[1], max_frames):
                    output = self.model(inputs=inference_view[:, i:i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self.model(inputs=inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            result = result[:, :input_frame_count] # -> (B, T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 1, 3, 4, 2) # (B, T, H, W, C)
            restored_videos = [list(torch.unbind(video_result, 0)) for video_result in torch.unbind(result, 0)]
            for restored_video in restored_videos:
                output_frame_count = len(restored_video)
                output_frame_shape = restored_video[0].shape
                assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return restored_videos
