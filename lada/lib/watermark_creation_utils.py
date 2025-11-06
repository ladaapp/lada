# SPDX-FileCopyrightText: Lada Authors
# SPDX-FileCopyrightText: Watermarking Detection - DNN Authors
# SPDX-License-Identifier: AGPL-3.0
# Code vendored from: https://github.com/tgenlis83/dnn-watermark

import os
import random
import subprocess
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_fonts(lang="en") -> List[str]:
    cmd_result = subprocess.run(["bash", "-c", f"fc-list :lang={lang} | cut -d : -f 1 | grep .ttf"], stdout=subprocess.PIPE)
    stdout = cmd_result.stdout.decode('utf-8').strip()
    font_paths = stdout.split(sep='\n') if stdout else []
    return font_paths

def generate_random_japanese_string(length: int) -> str:
    kanji: str = "日�会人年大十二本中長出三同�事自行社見月分議後前民生�発間対上部東�地合�内相方四定今回新場金員九入選立開手米力学問高代明実円関決子動京全目表戦経��氏現理調体化田当八六約主題下首意法不来作�要用制治度務強気小七成期公持野協取都和統以機平総加山�話世受区領多県続�安設保改数記院女初北午指権心界支第産結百派�報済書府活原先共得解名交資予川向際査勝面委告軍文反元重近千�認画海�利組知案道信策集在件団別物側任引使�次水半品昨論計死官増�特情投示変打男基私各始島直両朝革価式確村提運�果西勢減台広容必応演電歳住争談能無再位置企真流格有疑口過�放税検藤町常校料沢裁状工建語球営空職証土与��供可役構木割聞身費付施切由説転食比難防補車優夫研収断井何南石足違消境神番規術護展態導鮮�害配副算視条幹独警宮究�輸訪楽起万�店述残想線率病農州武声質念待試族象�助労例衛然早張映限親額監環験追審商葉義伝働形景落欧担好�賞訴辺�被株頭�毎医復仕去姿味負閣韓渡失移差衆�写評課末守若脳極種美岡影命含福蔵量望松非撃佐核観察整段横融型白深字答夜製票況音申様財港識注呼渉達"
    hiragana: str = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖ"
    katakana: str = "゠ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヷヸヹヺ・ーヽヾヿ"
    digits = '0123456789'
    punctuation: str = "、，゠＝…‥。『　』〜：！？♪"
    whitespace: str = " "
    characters: str = kanji + hiragana + katakana + digits + punctuation + whitespace
    result_str: str = "".join(random.choice(characters) for _ in range(length))
    return result_str

def generate_random_string(length: int) -> str:
    """
    Generate a random string of a given length

    Args:
        length (int): The length of the string

    Returns:
        str: The random string
    """
    alphabet_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digits = '0123456789'
    punctuation: str = "/+-_@#.!?,;:~'><="
    whitespace: str = " "
    characters: str = alphabet_lowercase + alphabet_uppercase + digits + punctuation + whitespace
    result_str: str = "".join(random.choice(characters) for _ in range(length))
    return result_str


def _get_position(
    img_width: int, img_height: int, text_size: int
) -> dict[str, Tuple[float, float]]:
    """
    Returns a random position for the watermark

    Args:
        img_width (int): The width of the image
        img_height (int): The height of the image
        text_size (int): The size of the text

    Returns:
        tuple: The position of the watermark
    """
    padding = 0.04
    positions = [
        {"top_left": (img_width * padding, img_height * padding + text_size)},
        {"top_right": (img_width * (1 - padding), img_height * padding + text_size)},
        {"bottom_left": (img_width * padding, img_height * (1 - padding) - text_size)},
        {
            "bottom_right": (
                img_width * (1 - padding),
                img_height * (1 - padding) - text_size,
            )
        },
        {"middle": (img_width // 2, img_height // 2)},  # middle
    ]
    return np.random.choice(positions)


def _get_rotation_from_position(position: dict) -> int:
    """
    Returns a rotation angle from a position
    TODO: TO FIX
    """
    rotations = []
    pos_key = list(position.keys())[0]
    if pos_key == "top_left":
        rotations = [0, 45, 90]
    elif pos_key == "top_right":
        rotations = [90, 135, 180]
    elif pos_key == "bottom_left":
        rotations = [0, 270, 315]
    elif pos_key == "bottom_right":
        rotations = [180, 225, 270]
    elif pos_key == "middle":
        rotations = [0]
    return 0  # np.random.choice(rotations)


def _get_alpha(min=0.25, max=0.9) -> int:
    """
    Returns a random alpha value for the watermark

    Args:
        position (dict): The position of the watermark
        rotation (int): The rotation angle

    Returns:
        int: The alpha value
    """
    alpha_min, alpha_max = 255 * min, 255 * max
    return np.random.randint(alpha_min, alpha_max)


def _get_color() -> tuple:
    """
    Returns a random color
    """
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        _get_alpha(),
    )


def get_direction_anchor_from_position(position: dict) -> Tuple[str, str]:
    """
    Returns the direction and anchor for the text

    Args:
        position (dict): The position of the text

    Returns:
        str: The direction and anchor
    """
    pos_key: str = list(position.keys())[0]

    if pos_key == "top_left":
        return "ltr", "lt"
    elif pos_key == "top_right":
        return "ltr", "rt"
    elif pos_key == "bottom_left":
        return "ltr", "lb"
    elif pos_key == "bottom_right":
        return "ltr", "rb"
    elif pos_key == "middle":
        return "ltr", "mm"
    return "ltr", "lt"


def _convert_bbox_to_correct_format(bbox: tuple | None) -> tuple:
    """
    Convert (x1, y1, x2, y2) to (x, y, w, h) with x, y the center of the bbox and w, h the width and height
    """
    if bbox is None:
        return None
    x = (bbox[0] + bbox[2]) // 2
    y = (bbox[1] + bbox[3]) // 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return x, y, w, h


def add_text_watermark(
    img: Image.Image, font_name: str, size=512, lang="en"
) -> Tuple[Image.Image, tuple | None, int]:
    """
    Add a text watermark to an image

    Args:
        img (Image.Image): The image
        font_name (str): The font name

    Returns:
        tuple: The image with the watermark, the bbox, the category

    """

    img = resize_image(img, size, size)
    w, h = img.size

    txt: str = generate_random_japanese_string(np.random.randint(4, 16)) if lang == "ja" else generate_random_string(np.random.randint(8, 24))
    size: int = np.random.randint(22, min(w, h) // 5)
    position: dict = _get_position(w, h, size)
    rotation: int = _get_rotation_from_position(position)
    color: tuple = _get_color()
    position_values: tuple[float, float] = position[list(position.keys())[0]]
    font = ImageFont.truetype(font_name, size=size, encoding='UTF-8')

    stroke = random.random() < 0.2
    stroke_width = np.random.randint(2, size//8) if stroke else 0
    stroke_fill = _get_color() if stroke else 0

    new_img = img.copy().convert("RGBA")
    txt_new_img = Image.new("RGBA", new_img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(txt_new_img)
    direction, anchor = get_direction_anchor_from_position(position)

    color_background = random.random() < 0.2
    if color_background:
        border_size = np.random.randint(0, size//4)
        background_color = _get_color()
        left, top, right, bottom = draw.textbbox(position_values, txt, font=font, direction=direction, anchor=anchor, stroke_width=stroke_width)
        draw.rectangle((left - border_size, top - border_size, right + border_size, bottom + border_size), fill=background_color)
    draw.text(
        position_values, txt, fill=color, font=font, direction=direction, anchor=anchor, stroke_width=stroke_width, stroke_fill=stroke_fill
    )  # pyre-ignore[6]

    txt_new_img = txt_new_img.rotate(rotation)
    combined = Image.alpha_composite(new_img, txt_new_img)

    bbox = txt_new_img.getbbox()
    bbox = _convert_bbox_to_correct_format(bbox)
    return combined.convert("RGB"), bbox, 2


def remove_background(img_in: Image.Image):
    """
    Utility function to remove the background of a logo
    """
    img = np.array(img_in)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # if logo is already grayscale
        gray = img
    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    # create an array full of 255, same size of original image
    white_mask = np.full(mask.shape, 255, dtype=np.uint8)
    mask = white_mask - mask
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(
        mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT
    )
    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)
    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    return Image.fromarray(result, "RGBA")


def _get_position_for_logo(
    img_width: int, img_height: int, logo_height: int, logo_width: int
) -> dict[str, Tuple[float, float]]:
    """
    Returns a position between:
    - top left
    - top right
    - bottom left
    - bottom right
    - middle
    With a padding of 10% of the image size
    """
    padding = 0.05

    positions = [
        {"top_left": (img_width * padding, img_height * padding)},  # top left
        {
            "top_right": (img_width * (1 - padding) - logo_width, img_height * padding)
        },  # top right
        {
            "bottom_left": (
                img_width * padding,
                img_height * (1 - padding) - logo_height,
            )
        },  # bottom left
        {
            "bottom_right": (
                img_width * (1 - padding) - logo_width,
                img_height * (1 - padding) - logo_height,
            )
        },  # bottom right
        {
            "middle": (
                img_width // 2 - logo_width // 2,
                img_height // 2 - logo_height // 2,
            )
        },  # middle
    ]

    return np.random.choice(positions)


def add_logo_watermark(img: Image.Image, logo: Image.Image, size=512) -> tuple:
    """
    Main to function to add logo watermark to an image
    Args:
        img (Image.Image): image to add watermark to
        logo (Image.Image): logo to add
    Returns:
        tuple: combined image, logo bbox, category
    """

    img = resize_image(img, size, size)
    w, h = img.size

    # logo = remove_background(logo)
    max_logo_size = np.random.uniform(0.1, 0.6)

    scale_factor: int = (
        max_logo_size / max(logo.width, logo.height) * min(img.width, img.height)
    )

    logo_resized = logo.resize(
        (int(logo.width * scale_factor), int(logo.height * scale_factor))
    )

    position: dict = _get_position_for_logo(
        w, h, logo_resized.height, logo_resized.width
    )

    rotation: int = _get_rotation_from_position(position)
    position_values = position[list(position.keys())[0]]
    position_values = int(position_values[0]), int(position_values[1])
    alpha = _get_alpha()

    logo_resized = logo_resized.rotate(rotation)
    logo_resized = np.array(logo_resized)
    alpha_channel = logo_resized[:, :, 3]
    alpha_channel[alpha_channel > 0] = alpha
    logo_resized[:, :, 3] = alpha_channel
    logo_resized = Image.fromarray(logo_resized, "RGBA")

    logo_transformed = Image.new("RGBA", img.size, (0, 0, 0, 0))
    logo_transformed.paste(logo_resized, position_values)
    bbox = logo_resized.getbbox()

    new_img = img.copy().convert("RGBA")
    combined = Image.alpha_composite(new_img, logo_transformed)

    if bbox is None:
        return combined, None, 1

    bbox = (
        *position_values,
        position_values[0] + bbox[2],
        position_values[1] + bbox[3],
    )
    bbox = _convert_bbox_to_correct_format(bbox)

    return combined.convert("RGB"), bbox, 1


def resize_image(image: Image, width: int, height: int) -> Image:
    """
    Resize an image to a specific size.

    Args:
        image (Image): The image to resize.
        size (Tuple[int, int]): The size to resize the image to.

    Returns:
        Image: The resized image.
    """
    image = cv2.resize(np.array(image), (width, height))
    return Image.fromarray(image)


def convert_to_yolo(
    file_name: str,
    bbox: Tuple[float],
    category_id: int,
    yolo_labels_path: str,
    yolo_images_path: str,
    watermarked_image: Image,
) -> None:
    label_annotation_filename: str = f"{file_name.split('.')[0]}.txt"

    if not os.path.exists(yolo_labels_path):
        os.makedirs(yolo_labels_path)

    if not os.path.exists(yolo_images_path):
        os.makedirs(yolo_images_path)

    with open(f"{yolo_labels_path}/{label_annotation_filename}", "w") as f:
        x = bbox[0] / watermarked_image.width
        y = bbox[1] / watermarked_image.height
        w = bbox[2] / watermarked_image.width
        h = bbox[3] / watermarked_image.height
        class_id = category_id - 1
        f.write(f"{class_id} {x} {y} {w} {h}\n")

    watermarked_image.save(f"{yolo_images_path}/{file_name}")