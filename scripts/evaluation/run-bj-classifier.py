# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import time
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.models import resnet50

from lada.utils.video_utils import read_video_frames

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(save_path, model_version='pov'):
    model = resnet50(pretrained=False)
    num_features = model.fc.in_features
    out_features = 3 if model_version == 'pov' else 2
    model.fc = torch.nn.Linear(num_features, out_features)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    device = torch.device("cuda:0")
    model = model.to(device)
    return model

def inference(model, imgs, device="cuda:0", batch_size=16, model_version='pov'):
    samples_positive_min = 0.7
    label_mappings_non_pov_model = {
        'bj': 0,
        'nonbj': 1
    }
    label_mappings_pov_model = {
        'bj_other': 0,
        'bj_pov': 1,
        'nonbj': 2
    }
    label_mappings = label_mappings_pov_model if model_version == 'pov' else label_mappings_non_pov_model

    batches = [imgs[x:x + batch_size] for x in range(0, len(imgs), batch_size)]
    results = {k: 0 for k in label_mappings.keys()}
    for sample_img_batch in batches:
        model_input = [transform(img) for img in sample_img_batch]
        model_input = torch.stack(model_input, dim=0)
        model_input = model_input.to(device)
        with torch.no_grad():
            model_output = model.forward(model_input)
        _, predictions = torch.max(model_output, 1)
        #print(f"model predictions: {predictions}")
        for label in label_mappings.keys():
            results[label] += torch.sum(predictions == label_mappings[label])

    predicted_label, predicted_label_count = max(results.items(), key=lambda i: i[1])
    predicted_label_count = predicted_label_count.cpu().item()
    confidence = predicted_label_count / len(imgs)
    return predicted_label, confidence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--model-version', type=str, default='pov')
    parser.add_argument('--model-path', type=str, default='experiments/bj_classifier/run1/checkpoint_13.pt')
    args = parser.parse_args()
    return args

def test_inference():
    import os
    os.environ['QT_QPA_PLATFORM'] = "xcb"
    file_path_inference = "bj_positive_sample.mp4"
    model_version = 'pov'
    model = load_model('./pov_bj_model.pt', model_version)
    frames = read_video_frames(file_path_inference, float32=False)
    start = time.perf_counter()
    predicted_label, confidence = inference(model, frames, model_version=model_version)
    print(f"Model predictions for file ${file_path_inference}: {predicted_label}, confidence: {confidence:0.2f}, runtime: {time.perf_counter()-start:.4f} seconds")


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_path, args.model_version)
    input_path = Path(args.input)
    if input_path.is_file():
        frames = read_video_frames(str(input_path), float32=False)
        result = inference(model, frames, model_version=args.model_version)
        print(f"{args.input},{result}")
    elif input_path.is_dir():
        for file_index, dir_entry in enumerate(input_path.iterdir()):
            if dir_entry.is_file():
                frames = read_video_frames(str(dir_entry), float32=False)
                predicted_label, confidence = inference(model, frames, batch_size=64)
                print(f"{str(dir_entry)},{predicted_label},{confidence:0.2f}")
