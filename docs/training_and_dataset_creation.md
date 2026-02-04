# Requirements
In order to work on the models and datasets you'll have to install the requirements:

1) Install everything mentioned in [Linux Install](linux_install.md)

2) Install python dependencies
   
   ```bash
   uv sync --group dev --inexact
   ````

3) Apply patches

    In order to fix resume training of the mosaic restoration model apply the following patch:
    ```bash
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/adjust_mmengine_resume_dataloader.patch
    ```

4) Download model weights

   To train the models and create your own datasets you'll also need these files
   ```shell
   wget -P model_weights/3rd_party/ 'https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
   wget -P model_weights/3rd_party/ 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
   wget -P model_weights/3rd_party/ 'https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth'
   wget -P model_weights/ 'https://huggingface.co/ladaapp/lada/resolve/main/lada_nsfw_detection_model_v1.3.pt?download=true'
   wget -P model_weights/ 'https://huggingface.co/ladaapp/lada/resolve/main/lada_watermark_detection_model_v1.3.pt?download=true'
   wget -P model_weights/3rd_party/ 'https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt'
   wget -P model_weights/3rd_party/ 'https://github.com/ORB-HD/deface/raw/refs/tags/v1.5.0/deface/centerface.onnx'
   wget -P model_weights/3rd_party/ 'https://huggingface.co/HoyerChou/BPJDet/resolve/main/ch_head_s_1536_e150_best_mMR.pt?download=true'
   ```

> [!CAUTION]
> The last download command currently may not work as the NudeNet project is set to age-restricted.
> You'll have to be logged into GitHub, then you can download the file manually on their [release page]('https://github.com/notAI-tech/NudeNet/releases/): release `v3.4` / file `640m.pt`
> The model is optional though and only used in `create-mosaic-restoration-dataset.py`.
   

# Training
The app uses two models: mosaic detection and mosaic restoration.
The goal of the mosaic detection model is to detect for each frame of the video if and where pixelated/mosaic regions exist.
It will try to crop and cut small clips and hand them over to the mosaic restoration model. This will try to recover what it can from those degraded frames and come up with a somewhat plausible replacement for those images.
These restored clip will replace the original content when they're reassembled with the original frames.

There is also a NSFW detection model used only to create the datasets for the other two models.

The following sections describe how to train and create a dataset for each model.
If you're not interested in training specific models you can use the pretrained model weights from Lada where needed.

> [!TIP]
> To gather source material for your dataset [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [gallery-dl](https://github.com/mikf/gallery-dl) are your friends.
> 
> To find and remove duplicate files I can recommend the tools [Czkawka](https://github.com/qarmin/czkawka) and [cbird](https://github.com/scrubbbbs/cbird).

> [!TIP]
> The scripts `scripts/dataset_creation/extract-video-frames.py` can be useful for creating image datasets.

> [!TIP]
> In some scripts OpenCV QT-based GUI features are using for debugging purposes. 
> OpenCV currently does not support wayland, you may want to force QT to use its X11 backend by setting the following environment variable
> `export QT_QPA_PLATFORM=xcb`.

## Mosaic restoration model
Before we can train the model we'll need to create a dataset.
AFAIK, there are no publicly available datasets for such purpose and I'll not provide one either. But you can create your own dataset for training mosaic removal models with the following procedure:

```shell
python scripts/dataset_creation/create-mosaic-restoration-dataset.py --input <input dir> --output-root <output dir>
```
Here `<input dir>` should be a directory containing your source material (adult video files without mosaics).

The script will detect regions of NSFW material and cut and crop short clips to be used for mosaic restoration training.

The script doesn't necessarily need to be called with additional options, but you should check them out (`--help`) and play with it to understand what's going on before running this on a lot of data.

For example:
You can optimize worker and memory limits according to your machine. You can also run the script in parallel on different subset of data using different GPUs.
There are options to create mosaic clips as well which can be useful to inspect generated mosaic clips.
Depending on your source material use the `--stride-length` option to prevent sampling too many scenes from the same (long) files.

Additional metadata and filtering can be adjusted as well. Check-out the *filter* / *add-metadata* switches.

Try it on a small subset of your data first to see how it works.
Also, check out the code `MosaicVideoDataset` in `mosaic_video_dataset.py` as well as the dataloader/dataset settings in `mosaic_restoration_generic_stage{1,2}.py` in the `config` dir to understand how this generated dataset will be used in training.

With the default settings only cropped NSFW scenes and their segmentation masks will be generated (two directories). Mosaics will gen generated on-the-fly while training by torch dataset `MosaicVideoDataset`.
There are options though to generate full-size scenes as well as mosaic scenes. I would not suggest to enable these options on your final large-sized dataset as it will take a lot of time and storage space.

The script uses the NSFW detection model. Some filtering options are available in the training config files and in the script for automatic filtering of invalid data.
Neither of those are perfectly accurate, and you probably want to validate and remove false-positive clips manually.

Have a look at the created metadata JSON files: You can find watermark (text,logos) detection and video quality information which may come in handy for filtering depending on your data sources.

> [!TIP]
> I used the following workflow for manual clean-up:
> (1) open the directory of created NSFW scenes in your file explorer in thumbnail view. (2) Wait until thumbnails have been created. (3) Based on what you can see in the thumbnail delete files if they contain watermarks or don't look like an actual NSFW scene. (4) write a shell script to delete corresponding mask and json metadata files 

Now, with a dataset at hand we're ready to train a model.

Training the mosaic restoration model is done in two steps. You can find training scripts in the project root directory and related configuration files in the `config` directory.
The first stage consists of training a BasicVSR++ model only with pixel loss (you'll need to create a dataset first)
```shell
python scripts/training/train-mosaic-restoration-basicvsrpp.py configs/basicvsrpp/mosaic_restoration_generic_stage1.py
```
> You can continue an interrupted run by adding `--resume` to the command line.

Before we can continue training stage2 you'll have to convert the trained weights into the GAN-compatible model with the following script
```shell
python scripts/training/convert-weights-basicvsrpp-stage1-to-stage2.py
```
Now we can continue training with additional GAN and perceptual losses.
```shell
python scripts/training/train-mosaic-restoration-basicvsrpp.py configs/basicvsrpp/mosaic_restoration_generic_stage2.py --load-from experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_10000_converted.pth
```

If you're happy with the model you can export it for inference and remove the discriminator model via:
```shell
python scripts/training/export-weights-basicvsrpp-stage2-for-inference.py
```

Note that the model is implemented in the MMagic / MMEngine framework. If you need to adjust model or training parameters you can do that by adjusting
the files in the `config` directory.

I'd recommend to read through [MMengine documentation](https://mmengine.readthedocs.io/en/latest/) first if you're not familiar with that library.

> [!TIP]
> In case you want to fine-tune you can find the full model weights on [HuggingFace](https://huggingface.co/ladaapp/lada)

## NSFW detection model
The purpose of this dataset is to train an image segmentation model which we can feed video frames to detect if and where NSFW regions exist in the image (binary segmentation task).
This model is then only used in the mosaic restoration and mosaic detection dataset creation scripts.
Creating this dataset is a labor-intensive process, you'll have to hand-label each image and draw segmentation masks around all NSFW regions visible in each image.

I used and recommend the tool [labelme](https://github.com/wkentaro/labelme) for this task.

Setup and use another virtual environment to avoid dependency conflicts with Lada dependencies when installing labelme
```shell
python -m venv .venv_labelme
source .venv_labelme/bin/activate
pip install labelme
```

Create training and validation directories for your NSFW detection dataset and fill it with diverse set of NSFW video frames and images
```shell
mkdir -p datasets/nsfw_detection_labelme/{train,val}
```

Then start the tool with this command

```shell
labelme --flags sfw --labels nsfw --nodata --autosave datasets/nsfw_detection_labelme/train
```

I'd suggest to keep the masks relatively tight around each object. In practise mosaics are often not very precise and also cover a good amount of SFW content around NSFW parts.
We'll randomly extend those masks automatically when working on mosaic detection or mosaic restoration so we should label them relatively precisely in this step.

In *labelme* use the *Draw Polygon* tool to label NSFW regions in the image.

<img src="../assets/screenshot_labelme_nsfw.png" alt="screenshot showing LabelMe program using polygon tool to label NSFW regions" height="300px">

You should also add some images without any visible NSFW content. For those, create a *SFW* label and no Polygons.

<img src="../assets/screenshot_labelme_sfw.png" alt="screenshot showing LabelMe program labeling image as SFW" height="300px">

Then continue this process with a couple more images.

Now let's train the model

> [!NOTE]
> To continue switch back to lada virtual environment `.venv`. Use the `.venv_labelme` only when annotating files in labelme.

We're training a YOLO v11 segmentation model (`yolo11m-seg`).
YOLO or rather the library we're using to train it (*ultralytics*) does not support labelme format so we'll have to convert into the format they can understand:
```shell
mkdir -p datasets/nsfw_detection/{train,val}/{images,labels}
python scripts/dataset_creation/convert-dataset-labelme-to-yolo.py --dir-in datasets/nsfw_detection_labelme/train --dir-out-images datasets/nsfw_detection/train/images --dir-out-labels datasets/nsfw_detection/train/labels
python scripts/dataset_creation/convert-dataset-labelme-to-yolo.py --dir-in datasets/nsfw_detection_labelme/val --dir-out-images datasets/nsfw_detection/val/images --dir-out-labels datasets/nsfw_detection/val/labels
```

With that step out of the way we're now ready to start the training process. Simply run
```shell
python scripts/training/train-nsfw-detection-yolo.py
```

> [!TIP]
> Under the hood, `ultralytics` package is used to train the model. You can read their [documentation](https://docs.ultralytics.com/) for further details.


Once that is done test it on some real-world NSFW videos (not from the source material you've been training with) using the following script:
```shell
python scripts/evaluation/view-yolo.py --input <path to your nsfw file> --model-path experiments/yolo/segment/train_nsfw_detection_yolo11m/weights/best.pt --screenshot-dir datasets/nsfw_detection_labelme/train
```

This will open a very simple GUI where you can seek through the video frame-by-frame to check the detection result (masks and prediction confidence levels).
If you find frames with false-positives or other parts the model incorrectly classified hit the `S` key to take a screenshot and save the frame in the specified directory.

<img src="../assets/screenshot_view_yolo.png" alt="screenshot showing view-yolo.py tool" width="45%">

Then you can fire up labelme once more and annotate this file properly.

Repeat these steps to annotate, convert, train, validate a couple of times, and you have built yourself a NSFW detection model.

> [!TIP]
> After the first training round you can run the model on some other files, convert the predictions from YOLO to labelme and validate and correct them in labelme before adding them to the dataset

## Mosaic detection model
Purpose of the mosaic detection model is to detect if and where mosaic regions are in a given video frame. It's used as the first part of the restoration pipeline of Lada.
Assuming you have a directory of NSFW videos and a trained NSFW detection model (either the pretrained model weights from lada or you've trained your own version following above-mentioned procedure)
the creation of this dataset will be an automatic process.

You can create a dataset for mosaic detection using the following command:

```shell
python create-mosaic-detection-dataset.py --input-root <path to source material / nsfw images>  --output-root <path to dataset dir>
```

The script will run the NSFW detection model on the source material. For frames with NSFW content it will use the segmentation mask and the original video frame
to create a new image where the NSFW region is replaced with a mosaic / pixelize pattern. This image as well as the segmentation mask of the mosaic is then saved in the output root directory.

The mosaic detection model will also be trained using ultralytics / YOLO library so the script will also convert the segmentation masks to YOLO-compatible labels.

The script comes with some options you may want to check and adjust before you're running this on a huge dataset.

> [!TIP]
> You can re-use the source material used for mosaic restoration or NSFW detection dataset creation.

For the model to also learn to differentiate between NSFW mosaics (which should be restored) and other mosaics (most commonly on faces) you will have to add such samples to your dataset.
I havn't trained a custom model/dataset for such purpose, instead I've used [Retinaface](https://github.com/yakhyo/retinaface-pytorch) for face detection and [Single-Human-Parsing-LIP](https://github.com/Yukun-Huang/Single-Human-Parsing-LIP) for head detection (filtering for classes hair and face) to create binary masks.
You'll just have to replace NsfwImageDetector detection/mask creation in the dataset creation script with inference code of these tools. Be prepared to clean up those samples manually though, these models are not very accurate given NSFW video data.

Now we can train the model via
```shell
python scripts/training/train-mosaic-detection-yolo.py
```

> [!TIP]
> Under the hood, `ultralytics` package is used to train the model. You can read their [documentation](https://docs.ultralytics.com/) for further details.

To check its performance you can use the `view-yolo.py` script as described in the training section of the NSFW detection model.

## Watermark detection model
This model is used to filter out scenes detected by the NSFW model obstructed by watermarks, text or logos. Otherwise, we could introduce unwanted artifacts when
applying mosaics on them when added to the mosaic restoration dataset.

After you collected NSFW images (and cleaned them so they do *not* contain any watermarks/logos/text) as well as some logos you can create - you guessed it - another YOLO dataset.

SFW images also do the trick but would suggest to keep the majority of files NSFW, e.g. you could throw in a few thousand COCO images which shouldn't contain any watermarks as far as I could see.

For the logos I'd suggest to get .png files with proper alpha channels for background separation. This way you can easily crop them to bounding rectangle to create more accurate box labels.

The following script automatically creates a dataset by applying the given logos on-top of the given images as well as pasting some random text over it.

The script will randomly choose from all available truetype fonts on your system. So I would suggest you install a few more and also include some "creative fonts" which may appear in real-world text/watermarks.

```shell
python scripts/dataset_creation/create-watermark-detection-dataset.py --train-images-dir <train images input dir> --val-images-dir <val images input dir> --logos-dir <logos input dir> --yolo-dir datasets/watermark_detection
```

After creating the dataset you can train it via
```shell
python scripts/training/train-watermark-detection-yolo.py
```
