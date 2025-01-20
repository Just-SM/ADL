import torch.optim as optim
from tqdm import tqdm
import torch

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataloaders.PaintingDatasets import CreatePaintingsDataLoaders

from models.MulitTaskModel import evaluate, MultiTaskModel


DATA_FOLDER = "data"
IMAGE_RAW_DATA = "raw_images"
DATA_CSV = "art_data_loaded.csv"
EPOCHS = 10


train_dataset, val_dataset, weights, num_of_classes = CreatePaintingsDataLoaders(
    os.path.join(
        DATA_FOLDER,
        DATA_CSV,
    ),
    data_folder=os.path.join(
        DATA_FOLDER,
        IMAGE_RAW_DATA,
    ),
)

style_weights, date_weights, type_weights = weights
style_classes, date_classes, type_classes = num_of_classes

# Model
model = MultiTaskModel(
    num_classes_style=len(style_classes),
    num_classes_date=len(date_classes),
    num_classes_type=len(type_classes),
).to(device)

# Loss Functions with Class Weights
criterion_style = torch.nn.CrossEntropyLoss(weight=style_weights.to(device))
criterion_date = torch.nn.CrossEntropyLoss(weight=date_weights.to(device))
criterion_type = torch.nn.CrossEntropyLoss(weight=type_weights.to(device))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_style_accuracy_1 = 0  # Track the best Top-1 accuracy for style
num_epochs = EPOCHS
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_dataset, total=len(train_dataset)):
        images = images.to(device)
        labels_style = labels["style"].to(device).long()
        labels_date = labels["date"].to(device).long()
        labels_type = labels["type"].to(device).long()

        optimizer.zero_grad()

        # Forward pass
        style_out, date_out, type_out, emb = model(images)

        # Compute losses
        loss_style = criterion_style(style_out, labels_style)
        loss_date = criterion_date(date_out, labels_date)
        loss_type = criterion_type(type_out, labels_type)

        loss = loss_style + loss_date + loss_type
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate the model on the validation set
    (
        style_accuracy_1,
        date_accuracy_1,
        type_accuracy_1,
        style_accuracy_3,
        date_accuracy_3,
        type_accuracy_3,
    ) = evaluate(model, val_dataset)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f}"
    )
    print(f"Validation Top-1 Accuracy for Style: {style_accuracy_1:.2f}%")
    print(f"Validation Top-1 Accuracy for Date: {date_accuracy_1:.2f}%")
    print(f"Validation Top-1 Accuracy for Type: {type_accuracy_1:.2f}%")

    # Save the model if the validation Top-1 accuracy for style improves
    if style_accuracy_1 > best_style_accuracy_1:
        best_style_accuracy_1 = style_accuracy_1
        torch.save(
            model.state_dict(),
            os.path.join(
                DATA_FOLDER,
                "model.pth",
            ),
        )
        print("Model saved with improved style accuracy.")
