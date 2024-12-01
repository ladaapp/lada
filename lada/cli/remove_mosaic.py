import argparse
from tqdm import tqdm
import os
import tempfile
from lada import MODEL_WEIGHTS_DIR, VERSION
from lada.lib.restored_mosaic_frames_generator import load_models, FrameRestorer
from lada.lib.video_utils import get_video_meta_data, VideoWriter
from lada.lib import audio_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to pixelated video file')
    parser.add_argument('--output', type=str, help='path to store restored video')
    parser.add_argument('--device', type=str, default="cuda:0", help='torch device to run the models on')
    parser.add_argument('--codec', type=str, default="h264", help='ffmpeg video codec. You probably want to use "h264" or "hevc" here')
    parser.add_argument('--crf', type=int, default=20, help='codec quality setting. The lower the better with the caveat of larger files size and CPU usage')
    parser.add_argument('--mosaic-restoration-model', type=str, default="basicvsrpp-generic")
    parser.add_argument('--mosaic-detection-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt'))
    parser.add_argument('--mosaic-restoration-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.1.pth'))
    parser.add_argument('--mosaic-restoration-config-path', type=str)
    parser.add_argument('--max-clip-length', type=int, default=180, help='number of consecutive frames that will be fed to mosaic restoration model. Lower values reduce RAM and VRAM usage. If set too low quality will reduce / flickering')
    parser.add_argument('--mosaic-cleaning',  default=False, action=argparse.BooleanOptionalAction, help='If enabled will clean detected mosaic pattern from noise before passing it to mosaic restoration model. Not recommended')
    parser.add_argument('--mosaic-cleaning-edge-detection-method', type=str, default="canny", help="either canny or pidinet")
    parser.add_argument('--mosaic-cleaning-edge-detection-model-path', type=str, help="path to pidinet tiny model")
    parser.add_argument('--preserve-relative-scale',  default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--moov-front',  default=False, action=argparse.BooleanOptionalAction, help="sets ffmpeg mov flags 'frag_keyframe+empty_moov+faststart'. Enables playing the output video while it's being written")
    parser.add_argument('--version',  action='store_true')

    args = parser.parse_args()
    return args


def cli():
    args = parse_args()
    if args.version:
        print("Lada: ", VERSION)
        return

    mosaic_cleaning_edge_detection_model_path = args.mosaic_cleaning_edge_detection_model_path if args.mosaic_cleaning and args.mosaic_cleaning_edge_detection_method == 'pidinet' else None

    mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, preferred_pad_mode = load_models(
        args.device, args.mosaic_restoration_model, args.mosaic_restoration_model_path, args.mosaic_restoration_config_path,
        args.mosaic_detection_model_path, mosaic_cleaning_edge_detection_model_path
    )

    video_metadata = get_video_meta_data(args.input)

    frame_restorer = FrameRestorer(args.device, args.input, args.preserve_relative_scale, args.max_clip_length, args.mosaic_restoration_model,
                 mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, preferred_pad_mode, mosaic_cleaning=args.mosaic_cleaning)

    video_tmp_file_output_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(os.path.splitext(args.output)[0])}.tmp{os.path.splitext(args.output)[1]}")
    video_writer = VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height, video_metadata.video_fps_exact, codec=args.codec, crf=args.crf, moov_front=args.moov_front, time_base=video_metadata.time_base)

    try:
        for restored_frame, restored_frame_pts in tqdm(frame_restorer(), total=video_metadata.frames_count, desc="Processing frames"):
            video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
    except (RuntimeError, StopIteration) as e:
        print("Video processing stopped. Output video may be incomplete")
        print("Exception caught : ")
        print(e)

    video_writer.release()

    audio_utils.combine_audio_video_files(args.input, video_tmp_file_output_path, args.output)


if __name__ == '__main__':
    cli()
