import argparse
import gettext
import locale
import mimetypes
import os
import sys
import pathlib
import tempfile
import textwrap
from gettext import gettext as _

import av
import torch
from tqdm import tqdm

from lada import MODEL_WEIGHTS_DIR, VERSION, DETECTION_MODEL_NAMES_TO_FILES, RESTORATION_MODEL_NAMES_TO_FILES, \
    get_available_restoration_models, get_available_detection_models
from lada.lib import audio_utils
from lada.lib.frame_restorer import load_models, FrameRestorer
from lada.lib.video_utils import get_video_meta_data, VideoWriter


class TranslatableHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
        super(TranslatableHelpFormatter, self).__init__(*args, **kwargs)

    def add_usage(self, usage, actions, groups, prefix=None):
        prefix = _("Usage: ")
        args = usage, actions, groups, prefix
        self._add_item(self._format_usage, args)

def setup_argparser() -> argparse.ArgumentParser:
    examples_header_text = _("Examples:")

    example1_text = _("Restore video with default settings:")
    example1_command = _("%(prog)s --input input.mp4")

    example2_text = _("Restore all videos found in the specified directory and save them to a different folder:")
    example2_command = _("%(prog)s --input path/to/input/dir/ --output /path/to/output/dir/")

    example3_text = _("Use a GPU-accelerated codec for encoding the restored video:")
    example3_command = _("%(prog)s --input input.mp4 --codec hevc_nvenc --crf 20")

    parser = argparse.ArgumentParser(
        usage=_('%(prog)s [options]'),
        description=_("Restore pixelated adult videos (JAV)"),
        epilog=_(textwrap.dedent(f'''\
            {examples_header_text}
                * {example1_text}
                    {example1_command}
                * {example2_text}
                     {example2_command}
                * {example3_text}
                    {example3_command}
            ''')),
        formatter_class=TranslatableHelpFormatter,
        add_help=False)

    group_general = parser.add_argument_group(_('General'))
    group_general.add_argument('--input', type=str, help=_('Path to pixelated video file or directory containing video files'))
    group_general.add_argument('--output', type=str, help=_('Path used to save output file(s). If path is a directory then file name will be chosen automatically (see --output-file-pattern). If no output path was given then the directory of the input file will be used'))
    group_general.add_argument('--output-file-pattern', type=str, default="{orig_file_name}.restored.mp4", help=_("Pattern used to determine output file name(s). Used when input is a directory or a file with no output path specified"))
    group_general.add_argument('--device', type=str, default="cuda:0", help=_('Device used for running Restoration and Detection models. Use "cpu" or "cuda". If you have multiple GPUs you can select a specific one via index e.g. "cuda:0" (default: %(default)s)'))
    group_general.add_argument('--list-devices', action='store_true', help=_("List available devices and exit"))
    group_general.add_argument('--version', action='store_true', help=_("Display version and exit"))
    group_general.add_argument('--help', action='store_true', help=_("Show this help message and exit"))

    export = parser.add_argument_group(_('Export (Encoder settings)'))
    export.add_argument('--codec', type=str, default="h264", help=_('FFmpeg video codec. E.g. "h264, "hevc" or "hevc_nvenc". Use "--list-codecs" to see whats available. (default: %(default)s)'))
    export.add_argument('--list-codecs', action='store_true', help=_("List available codecs and hardware devices / GPUs for hardware-accelerated video encoding"))
    export.add_argument('--crf', type=int, default=None, help=_('Constant rate factor (CRF). Quality setting of the video encoder. Lower values will result in higher quality but larger file sizes. If you have selected GPU codecs "h264_nvenc" or "hevc_nvenc" then the option "qp" will be used instead as those encoders don\'t support the "crf" option. (default: %(default)s)'))
    export.add_argument('--preset', type=str, default=None, help=_('Encoder preset. Mostly affects file-size and speed. (default: %(default)s)'))
    export.add_argument('--moov-front',  default=False, action=argparse.BooleanOptionalAction, help=_("Sets ffmpeg mov flags 'frag_keyframe+empty_moov+faststart'. Enables playing the output video while it's being written (default: %(default)s)"))
    export.add_argument('--custom-encoder-options', type=str, help=_("Pass arbitrary encoder options. Pass it like you'd specify them using ffmpeg. For example: --custom-encoder-options \"-rc-lookahead 32 -rc vbr_hq\". Official FFmpeg Codecs Documentation: https://ffmpeg.org/ffmpeg-codecs.html"))

    group_restoration = parser.add_argument_group(_('Mosaic Restoration'))
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default="basicvsrpp", help=_("Model used to restore mosaic clips (default: %(default)s)"))
    group_restoration.add_argument('--list-mosaic-restoration-models', action='store_true', help=_("List available restoration model weights found in MODEL_WEIGHTS_DIR and exit (default location is './model_weights' if not overwritten by environment variable MODEL_WEIGHTS_DIR)"))
    group_restoration.add_argument('--mosaic-restoration-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth'), help=_("Path to restoration model weights file (default: %(default)s)"))
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str, default=None, help=_("Path to restoration model configuration file"))
    group_restoration.add_argument('--max-clip-length', type=int, default=180, help=_('Maximum number of frames for restoration. Higher values improve temporal stability. Lower values reduce memory footprint. If set too low flickering could appear (default: %(default)s)'))

    group_detection = parser.add_argument_group(_('Mosaic Detection'))
    group_detection.add_argument('--mosaic-detection-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_fast.pt'), help=_("Path to restoration model weights file (default: %(default)s)"))
    group_detection.add_argument('--list-mosaic-detection-models', action='store_true', help=_("List available detection model weights found in MODEL_WEIGHTS_DIR and exit (default location is './model_weights' if not overwritten by environment variable MODEL_WEIGHTS_DIR)"))

    return parser

def filter_video_files(directory_path: str):
    video_files = []
    for name in os.listdir(directory_path):
        path = os.path.join(directory_path, name)
        if not os.path.isfile(path):
            continue
        mime_type, _ = mimetypes.guess_file_type(path)
        if not mime_type:
            continue
        if not mime_type.lower().startswith("video/"):
            continue
        video_files.append(path)
    return video_files

def get_output_file_path(input_file_path: str, output_directory: str, output_file_pattern: str):
    output_file_name = output_file_pattern.replace("{orig_file_name}", pathlib.Path(input_file_path).stem)
    return os.path.join(output_directory, output_file_name)

def dump_pyav_codecs():
    print(_("PyAV version:"))
    print(f"\t{av.__version__}")

    from lada.lib.video_utils import get_available_video_encoder_codecs
    print(_("Available video encoders:"))
    for short_name, long_name in get_available_video_encoder_codecs():
        print("\t%-18s %s" % (short_name, long_name))

    try:
        from av.codec.hwaccel import hwdevices_available
        print(_("Encoders with support for hardware acceleration (GPU):"))
        for x in hwdevices_available():
            print(f"\t{x}")
    except ImportError:
        print("Unable to list available hwdevices, ImportError")

    try:
        from av.codec.codec import dump_hwconfigs
        dump_hwconfigs()
    except ImportError:
        print("Unable to list hwdevice configs, ImportError")

def dump_torch_devices():
    device_header = _("Device")
    description_header = _("Description")
    s = _("Available devices:")
    s += f"\n\t{device_header}\t{description_header}"
    s += f"\n\t{len(device_header)*"-"}\t{len(description_header)*"-"}"
    s += "\n\tcpu\tCPU"
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_properties(i).name
        s += f"\n\tcuda:{i}\t{gpu_name}"
    print(s)

def dump_available_detection_models():
    s = _("Available detection models:")
    detection_model_names = get_available_detection_models()
    if len(detection_model_names) == 0:
        s += f"\n\t{_("None!")}"
    else:
        model_name_header = _("Name")
        model_path_header = _("Path")
        s += f"\n\t{model_name_header}\t{model_path_header}"
        s += f"\n\t{len(model_name_header) * "-"}\t{len(model_path_header) * "-"}"
        for name in detection_model_names:
            s += f"\n\t{name}\t{DETECTION_MODEL_NAMES_TO_FILES[name]}"
    print(s)

def dump_available_restoration_models():
    s = _("Available restoration models:")
    restoration_model_names = get_available_restoration_models()
    if len(restoration_model_names) == 0:
        s += f"\n\t{_("None!")}"
    else:
        model_name_header = _("Name")
        model_path_header = _("Path")
        s += f"\n\t{model_name_header}\t{model_path_header}"
        s += f"\n\t{len(model_name_header) * "-"}\t{len(model_path_header) * "-"}"
        for name in restoration_model_names:
            s += f"\n\t{name}\t{RESTORATION_MODEL_NAMES_TO_FILES[name]}"
    print(s)

def process_video_file(input_path: str, output_path: str, device, mosaic_restoration_model, mosaic_detection_model,
                       mosaic_restoration_model_name, preferred_pad_mode, max_clip_length, codec, crf, moov_front, preset, custom_encoder_options, print_prefix=""):
    video_metadata = get_video_meta_data(input_path)

    frame_restorer = FrameRestorer(device, input_path, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode)
    success = True
    video_tmp_file_output_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(os.path.splitext(output_path)[0])}.tmp{os.path.splitext(output_path)[1]}")
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    try:
        frame_restorer.start()

        with VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height,
                         video_metadata.video_fps_exact, codec=codec, crf=crf, moov_front=moov_front,
                         time_base=video_metadata.time_base, preset=preset,
                         custom_encoder_options=custom_encoder_options) as video_writer:
            for elem in tqdm(frame_restorer, total=video_metadata.frames_count, desc=_("Processing frames")):
                if elem is None:
                    success = False
                    print("Error on export: frame restorer stopped prematurely")
                    break
                (restored_frame, restored_frame_pts) = elem
                video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
    except (Exception, KeyboardInterrupt) as e:
        success = False
        if isinstance(e, KeyboardInterrupt):
            raise e
        else:
            print("Error on export", e)
    finally:
        frame_restorer.stop()

    if success:
        print(_("Processing audio"))
        audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_path)
    else:
        if os.path.exists(video_tmp_file_output_path):
            os.remove(video_tmp_file_output_path)

def setup_input_and_output_paths(input_arg, output_arg, output_file_pattern):
    single_file_input = os.path.isfile(input_arg)

    if single_file_input:
        input_files = [os.path.abspath(input_arg)]
    else:
        input_files = filter_video_files(input_arg)

    if len(input_files) == 0:
        print(_("No video files found"))
        sys.exit(1)

    if single_file_input:
        if not output_arg:
            input_file_path = input_files[0]
            output_dir_path = str(pathlib.Path(input_file_path).parent)
            output_files = [get_output_file_path(input_file_path, output_dir_path, output_file_pattern)]
        else:
            output_files = [output_arg]
    else:
        if output_arg:
            if not os.path.exists(output_arg):
                os.makedirs(output_arg)
            output_dir_path = output_arg
        else:
            output_dir_path = str(pathlib.Path(input_files[0]).parent)
        output_files = [get_output_file_path(input_file_path, output_dir_path, output_file_pattern) for input_file_path in input_files]

    assert len(input_files) == len(output_files)

    return input_files, output_files

def init_localization():
    APP_NAME = 'lada'
    LOCALE_DIR =  './translations'
    try:
        locale.bindtextdomain(APP_NAME, LOCALE_DIR)
        locale.textdomain(APP_NAME)
    except AttributeError as e:
        pass
        # TODO: Workaround for Windows as reported in #88
        #  Translations of .ui files will probably not work then
    gettext.bindtextdomain(APP_NAME, LOCALE_DIR)
    gettext.textdomain(APP_NAME)

def main():
    init_localization()
    argparser = setup_argparser()
    args = argparser.parse_args()
    if args.version:
        print("Lada: ", VERSION)
        sys.exit(0)
    if args.list_codecs:
        dump_pyav_codecs()
        sys.exit(0)
    if args.list_mosaic_detection_models:
        dump_available_detection_models()
        sys.exit(0)
    if args.list_mosaic_restoration_models:
        dump_available_restoration_models()
        sys.exit(0)
    if args.list_devices:
        dump_torch_devices()
        sys.exit(0)
    if args.help or not args.input:
        argparser.print_help()
        sys.exit(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(_("GPU {device} selected but CUDA is not available").format(device=args.device))
        sys.exit(1)
    if "{orig_file_name}" not in args.output_file_pattern or "." not in args.output_file_pattern:
        print(_("Invalid file name pattern. It must include the template string '{orig_file_name}' and a file extension"))
        sys.exit(1)
    if os.path.isdir(args.input) and args.output is not None and os.path.isfile(args.output):
        print(_("Invalid output directory. If input is a directory then --output must also be set to a directory"))
        sys.exit(1)
    if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
        print(_("Invalid input. No file or directory at {input_path}").format(input_path=args.input))
        sys.exit(1)

    mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
        args.device, args.mosaic_restoration_model, args.mosaic_restoration_model_path, args.mosaic_restoration_config_path,
        args.mosaic_detection_model_path
    )

    input_files, output_files = setup_input_and_output_paths(args.input, args.output, args.output_file_pattern)

    single_file_input = len(input_files) == 1

    for input_path, output_path in zip(input_files, output_files):
        if not single_file_input:
            print(f"{os.path.basename(input_path)}:")
        try:
            process_video_file(input_path=input_path, output_path=output_path, device=args.device, mosaic_restoration_model=mosaic_restoration_model, mosaic_detection_model=mosaic_detection_model,
                               mosaic_restoration_model_name=args.mosaic_restoration_model, preferred_pad_mode=preferred_pad_mode, max_clip_length=args.max_clip_length,
                               codec=args.codec, crf=args.crf, moov_front=args.moov_front, preset=args.preset, custom_encoder_options=args.custom_encoder_options)
        except KeyboardInterrupt:
            print(_("Received Ctrl-C, stopping restoration."))
            break

if __name__ == '__main__':
    main()
