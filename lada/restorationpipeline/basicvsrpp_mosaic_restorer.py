import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device, fp16, clip_length):
        self.model = model
        self.cpu_buffer = torch.empty(1, clip_length, 3, 256, 256, dtype=torch.uint8, device='cpu', pin_memory=True)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.inference_buffer = torch.empty(1, clip_length, 3, 256, 256, dtype=self.dtype, device=device, memory_format=torch.channels_last_3d)

    def restore(self, video: list[torch.Tensor], max_frames=-1) -> list[torch.Tensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        with torch.inference_mode():
            result = []
            cpu_buffer_view = self.cpu_buffer[0][:input_frame_count]
            inference_view = self.inference_buffer[:, :input_frame_count]

            torch.stack([x.permute(2, 0, 1) for x in video], dim=0, out=cpu_buffer_view)
            inference_view.copy_(cpu_buffer_view, non_blocking=True)
            inference_view.div_(255.0)

            if max_frames > 0:
                for i in range(0, input.shape[1], max_frames):
                    output = self.model(inputs=self.cpu_buffer[:, i:i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self.model(inputs=inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            result = result.squeeze(0)[:input_frame_count] # -> (T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
            result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result
