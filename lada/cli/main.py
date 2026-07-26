# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import multiprocessing
import os
import sys
import tempfile
import textwrap

# When frozen (PyInstaller) on macOS, multiprocessing spawn re-executes this
# executable with -B -S -I -c "from multiprocessing.resource_tracker import main; main(...)".
# PyTorch triggers this. Stdlib freeze_support() only handles worker processes
# (argv[1] == '--multiprocessing-fork'); it does not handle the resource_tracker
# process (which uses -c). See: Lib/multiprocessing/spawn.py `is_forking()` and
# `freeze_support()` in CPython.
# (https://github.com/python/cpython/blob/629a363ddd2889f023d5925506e61f5b6647accd/Lib/multiprocessing/spawn.py#L57-L80).
# Run the -c code and exit so argparse never sees it.
if getattr(sys, "frozen", False) and sys.platform == "darwin":
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-c" and i + 1 < len(sys.argv):
            exec(compile(sys.argv[i + 1], "<string>", "exec"))
            sys.exit(0)
        if sys.argv[i] == "-c":
            break

try:
    import torch
except ModuleNotFoundError:
    from lada import IS_FLATPAK
    if IS_FLATPAK:
        print(_("No GPU Add-On installed"))
        print(_("In order to use the application you need to install one of Lada's Flatpak Add-Ons that is compatible with your hardware"))
        sys.exit(1)
    else:
        raise

from lada import VERSION, ModelFiles, _get_log_dir
from lada.cli import utils
from lada.export.multi_device_scheduler import ExportEvent, ExportWorkerSettings, MultiDeviceExportScheduler, generate_run_id
from lada.export.single_file import process_video_file
from lada.utils import device_utils, video_utils
from lada.utils.device_utils import DeviceParseError
from lada.utils.os_utils import gpu_has_fp16_acceleration, get_default_torch_device
from lada.restorationpipeline import load_models
from lada.utils.video_utils import get_default_preset_name


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError(_("value must be greater than 0"))
    return parsed_value

def setup_argparser() -> argparse.ArgumentParser:
    examples_header_text = _("Examples:")

    example1_text = _("Restore video with default settings:")
    example1_command = _("%(prog)s --input input.mp4")

    example2_text = _("Restore all videos found in the specified directory and save them to a different folder:")
    example2_command = _("%(prog)s --input path/to/input/dir/ --output /path/to/output/dir/")

    example3_text = _("Use Nvidia hardware-accelerated encoder by selecting a preset:")
    example3_command = _("%(prog)s --input input.mp4 --encoding-preset hevc-nvidia-gpu-hq")

    example4_text = _("Set encoding parameters directly without using an encoding preset:")
    example4_command = _("%(prog)s --input input.mp4 --encoder libx265 --encoder-options '-crf 26 -preset fast -x265-params log_level=error'")

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
                * {example4_text}
                    {example4_command}
            ''')),
        formatter_class=utils.TranslatableHelpFormatter,
        add_help=False)

    group_general = parser.add_argument_group(_('General'))
    group_general.add_argument('--input', type=str, help=_('Path to pixelated video file or directory containing video files'))
    group_general.add_argument('--output', type=str, help=_('Path used to save output file(s). If path is a directory then file name will be chosen automatically (see --output-file-pattern). If no output path was given then the directory of the input file will be used'))
    group_general.add_argument('--temporary-directory', type=str, default=tempfile.gettempdir(), help=_('Directory for temporary video files during restoration process. Alternatively, you can use the environment variable TMPDIR. (default: %(default)s)'))
    group_general.add_argument('--output-file-pattern', type=str, default="{orig_file_name}.restored.mp4", help=_("Pattern used to determine output file name(s). Used when input is a directory, or a file but no output path was specified. Must include the placeholder '{orig_file_name}'. (default: %(default)s)"))
    group_general.add_argument('--device', type=str, default=get_default_torch_device(), help=_('Device used for running Restoration and Detection models. Use "--list-devices" to see what\'s available (default: %(default)s)'))
    group_general.add_argument('--devices', type=str, help=_('Devices used for parallel multi-file export. Examples: auto, cuda:0, cuda:0,cuda:1, cpu. If omitted, the existing --device behavior is used'))
    group_general.add_argument('--parallel', type=positive_int, help=_('Maximum number of files processed in parallel when --devices is used. Defaults to the number of selected worker slots'))
    group_general.add_argument('--jobs-per-device', type=positive_int, default=1, help=_('Number of parallel jobs per selected GPU. The recommended value for video restoration is 1. (default: %(default)s)'))
    group_general.add_argument('--gpu-worker-policy', choices=('auto', 'fixed', 'one-per-gpu'), help=_('How multi-GPU export sizes workers per GPU. auto uses VRAM estimates, fixed keeps --jobs-per-device, one-per-gpu forces one worker per GPU.'))
    group_general.add_argument('--worker-cpu-threads', type=positive_int, help=_('CPU threads per multi-device worker. Defaults to an automatic value based on CPU count and worker count'))
    group_general.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=gpu_has_fp16_acceleration(), help=_("Reduces VRAM usage and may increase speed on modern GPUs, with negligible quality difference. (default: %(default)s)"))
    group_general.add_argument('--list-devices', action='store_true', help=_("List available devices and exit"))
    group_general.add_argument('--version', action='store_true', help=_("Display version and exit"))
    group_general.add_argument('--help', action='store_true', help=_("Show this help message and exit"))

    export = parser.add_argument_group(_('Export'))
    export.add_argument('--encoding-preset', type=str, default=get_default_preset_name(), help=_('Select encoding preset by name. Use "--list-encoding-presets" to see what\'s available. Ignored if "--encoder" and "--encoder-options" are used (default: %(default)s)'))
    export.add_argument('--list-encoding-presets', action='store_true', help=_("List available encoding presets and exit"))
    export.add_argument('--encoder', type=str, help=_('Select video encoder by name. Use "--list-encoders" to see what\'s available. (default: %(default)s)'))
    export.add_argument('--list-encoders', action='store_true', help=_("List available encoders and exit"))
    export.add_argument('--encoder-options', type=str, help=_("Space-separated list of options for the encoder set via \"--encoder\". Use \"--list-encoder-options\" to see what's available. (default: %(default)s)"))
    export.add_argument('--list-encoder-options', metavar='ENCODER', type=str, help=_("List available options of the given encoder and exit"))
    export.add_argument('--mp4-fast-start',  default=False, action=argparse.BooleanOptionalAction, help=_("Allows playing the file while it's being written. Sets .mp4 mov flags \"frag_keyframe+empty_moov+faststart\". (default: %(default)s)"))

    group_restoration = parser.add_argument_group(_('Mosaic Restoration'))
    group_restoration.add_argument('--list-mosaic-restoration-models', action='store_true', help=_("List available restoration models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default='basicvsrpp-v1.2', help=_('Name of detection model or path to model weights file. Use "--list-mosaic-restoration-models" to see what\'s available. (default: %(default)s)'))
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str, default=None, help=_("Path to restoration model configuration file. You'll not have to set this unless you're training your own custom models"))
    group_restoration.add_argument('--max-clip-length', type=int, default=180, help=_('Maximum number of frames for restoration. Higher values improve temporal stability. Lower values reduce memory footprint. If set too low flickering could appear (default: %(default)s)'))

    group_detection = parser.add_argument_group(_('Mosaic Detection'))
    group_detection.add_argument('--mosaic-detection-model', type=str, default='v4-fast', help=_('Name of detection model or path to model weights file. Use "--list-mosaic-detection-models" to see what\'s available. (default: %(default)s)'))
    group_detection.add_argument('--list-mosaic-detection-models', action='store_true', help=_("List available detection models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_detection.add_argument('--detect-face-mosaics', action=argparse.BooleanOptionalAction, default=False, help=_("Detect and ignore areas of pixelated faces. Can prevent restoration artifacts but may worsen detection of NSFW mosaics. Available for models v3 and newer. (default: %(default)s)"))

    return parser

def should_use_multi_device_scheduler(input_file_count: int, worker_device_slots: list[str], devices_arg: str | None) -> bool:
    return devices_arg is not None and input_file_count > 1 and len(worker_device_slots) > 1


def print_scheduler_event(event: ExportEvent):
    if event.event_type == "worker_started":
        print(_("Worker {worker_id} started on {device}").format(worker_id=event.worker_id, device=event.device))
    elif event.event_type == "task_started":
        print(_("{filename}: started on {device}").format(filename=os.path.basename(event.input_path), device=event.device))
    elif event.event_type == "task_finished":
        print(_("{filename}: finished on {device}").format(filename=os.path.basename(event.input_path), device=event.device))
    elif event.event_type == "task_failed":
        print(_("{filename}: failed on {device}: {error}").format(filename=os.path.basename(event.input_path), device=event.device, error=event.error))
    elif event.event_type == "log":
        message = event.error or event.message
        if message:
            print(message)


def print_scheduler_summary(summary):
    print(_("Export summary: {success} succeeded, {failed} failed, total time {duration:.1f}s").format(
        success=summary.successful_count,
        failed=summary.failed_count,
        duration=summary.duration_seconds,
    ))
    if summary.failed_tasks:
        print(_("Failed files:"))
        for failed_task in summary.failed_tasks:
            print(f"  {failed_task.input_path}: {failed_task.error}")

def main():
    multiprocessing.freeze_support()
    argparser = setup_argparser()
    args = argparser.parse_args()
    if args.version:
        print("Lada: ", VERSION)
        sys.exit(0)
    if args.list_encoders:
        utils.dump_encoders()
        sys.exit(0)
    if args.list_mosaic_detection_models:
        utils.dump_available_detection_models()
        sys.exit(0)
    if args.list_mosaic_restoration_models:
        utils.dump_available_restoration_models()
        sys.exit(0)
    if args.list_devices:
        utils.dump_torch_devices()
        sys.exit(0)
    if args.list_encoding_presets:
        utils.dump_available_encoding_presets()
        sys.exit(0)
    if args.list_encoder_options:
        utils.dump_encoder_options(args.list_encoder_options)
        sys.exit(0)
    if args.help or not args.input:
        argparser.print_help()
        sys.exit(0)
    parsed_devices = None
    worker_device_slots = None
    if args.devices is None:
        try:
            selected_device = device_utils.validate_torch_device(args.device)
        except DeviceParseError as e:
            print(e)
            sys.exit(1)
    else:
        try:
            parsed_devices = device_utils.parse_devices_arg(args.devices)
            worker_device_slots = device_utils.build_worker_device_slots(
                parsed_devices,
                jobs_per_device=args.jobs_per_device,
                parallel=args.parallel,
                allow_parallel_cpu=args.parallel is not None,
            )
        except (DeviceParseError, ValueError) as e:
            print(e)
            sys.exit(1)

        if args.devices.strip().lower() == "auto":
            cuda_devices = [device for device in parsed_devices if device.startswith("cuda:")]
            if cuda_devices:
                print(_("Detected CUDA devices: {devices}").format(devices=", ".join(cuda_devices)))
            else:
                print(_("No CUDA devices detected. Falling back to {device}").format(device=parsed_devices[0]))
        selected_device = worker_device_slots[0]
    if "{orig_file_name}" not in args.output_file_pattern or "." not in args.output_file_pattern:
        print(_("Invalid file name pattern. It must include the template string '{orig_file_name}' and a file extension"))
        sys.exit(1)
    if os.path.isdir(args.input) and args.output is not None and os.path.isfile(args.output):
        print(_("Invalid output directory. If input is a directory then --output must also be set to a directory"))
        sys.exit(1)
    if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
        print(_("Invalid input. No file or directory at {input_path}").format(input_path=args.input))
        sys.exit(1)
    if args.temporary_directory and not os.path.isdir(args.temporary_directory):
        print(_("Temporary directory {temporary_path} doesn't exist. Creating…").format(temporary_path=args.temporary_directory))
        os.makedirs(args.temporary_directory)

    if detection_modelfile := ModelFiles.get_detection_model_by_name(args.mosaic_detection_model):
        mosaic_detection_model_path = detection_modelfile.path
    elif os.path.isfile(args.mosaic_detection_model):
        mosaic_detection_model_path = args.mosaic_detection_model
    else:
        print(_("Invalid mosaic detection model"))
        sys.exit(1)

    if restoration_modelfile := ModelFiles.get_restoration_model_by_name(args.mosaic_restoration_model):
        mosaic_restoration_model_name = args.mosaic_restoration_model
        mosaic_restoration_model_path = restoration_modelfile.path
    elif os.path.isfile(args.mosaic_restoration_model):
        mosaic_restoration_model_path = args.mosaic_restoration_model
        mosaic_restoration_model_name = 'basicvsrpp' # Assume custom model is basicvsrpp. DeepMosaics custom path is not supported
    else:
        print(_("Invalid mosaic restoration model"))
        sys.exit(1)

    encoder = None
    encoder_options = None
    if args.encoder:
        encoder = args.encoder
        encoder_options = args.encoder_options if args.encoder_options else ''
    elif args.encoding_preset:
        encoding_presets = video_utils.get_encoding_presets()
        found = False
        for preset in encoding_presets:
            if preset.name == args.encoding_preset:
                found = True
                encoder = preset.encoder_name
                encoder_options = preset.encoder_options
                break
        if not found:
            print(_("Invalid encoding preset"))
            sys.exit(1)
    else:
        print(_('Either "--encoding-preset" or "--encoder" together with "--encoder-options" must be used'))
        sys.exit(1)
    assert encoder is not None and encoder_options is not None

    input_files, output_files = utils.setup_input_and_output_paths(args.input, args.output, args.output_file_pattern)

    single_file_input = len(input_files) == 1

    if args.devices is not None and should_use_multi_device_scheduler(len(input_files), worker_device_slots, args.devices):
        worker_settings = ExportWorkerSettings(
            base_temp_dir=args.temporary_directory,
            run_id=generate_run_id(),
            mosaic_restoration_model_name=mosaic_restoration_model_name,
            mosaic_restoration_model_path=mosaic_restoration_model_path,
            mosaic_restoration_config_path=args.mosaic_restoration_config_path,
            mosaic_detection_model_path=mosaic_detection_model_path,
            fp16=args.fp16,
            detect_face_mosaics=args.detect_face_mosaics,
            max_clip_length=args.max_clip_length,
            encoder=encoder,
            encoder_options=encoder_options,
            mp4_fast_start=args.mp4_fast_start,
            cpu_threads_per_worker=args.worker_cpu_threads,
            log_directory=str(_get_log_dir()),
        )
        scheduler = MultiDeviceExportScheduler(
            input_files=input_files,
            output_files=output_files,
            devices=parsed_devices,
            settings=worker_settings,
            parallel=args.parallel,
            jobs_per_device=args.jobs_per_device,
            allow_parallel_cpu=args.parallel is not None,
            gpu_worker_policy=args.gpu_worker_policy,
        )
        print(_("Multi-device export enabled: {workers} workers").format(workers=scheduler.worker_count))
        try:
            summary = scheduler.run(event_callback=print_scheduler_event)
            print_scheduler_summary(summary)
        except KeyboardInterrupt:
            print(_("Received Ctrl-C, stopping restoration."))
            scheduler.cancel()
        return

    device = torch.device(selected_device)
    mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
        device, mosaic_restoration_model_name, mosaic_restoration_model_path, args.mosaic_restoration_config_path,
        mosaic_detection_model_path, args.fp16, args.detect_face_mosaics
    )

    for input_path, output_path in zip(input_files, output_files):
        if not single_file_input:
            print(f"{os.path.basename(input_path)}:")
        try:
            process_video_file(input_path=input_path, output_path=output_path, temp_dir_path=args.temporary_directory, device=device, mosaic_restoration_model=mosaic_restoration_model, mosaic_detection_model=mosaic_detection_model,
                               mosaic_restoration_model_name=mosaic_restoration_model_name, preferred_pad_mode=preferred_pad_mode, max_clip_length=args.max_clip_length,
                               encoder=encoder, encoder_options=encoder_options, mp4_fast_start=args.mp4_fast_start,
                               progress_bar_factory=utils.Progressbar)
        except KeyboardInterrupt:
            print(_("Received Ctrl-C, stopping restoration."))
            break

if __name__ == '__main__':
    main()
