# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import os

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = torchvision.datasets.ImageFolder(root="datasets/pov_bj_scene_detection/train", transform=transform)
test_data = torchvision.datasets.ImageFolder(root="datasets/pov_bj_scene_detection/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

model = torchvision.models.resnet50(pretrained=True)

# Replace the last layer to match our own classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0")
model = model.to(device)

num_epochs = 15

experiment_name = "run1"
experiment_root_dir = "../../experiments/bj_classifier"
experiment_dir = f"{experiment_root_dir}/{experiment_name}"
os.makedirs(experiment_dir)

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # Evaluate
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the test loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels.data)

    train_loss /= len(train_data)
    test_loss /= len(test_data)
    test_acc = test_acc.double() / len(test_data)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
    best_test_acc = test_acc
    save_path = f"{experiment_dir}/checkpoint_{epoch}.pt"
    torch.save(model.state_dict(), save_path)

