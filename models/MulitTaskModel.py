import torch.nn as nn
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_loader):
    model.eval()
    correct_style_1, correct_date_1, correct_type_1 = 0, 0, 0
    correct_style_3, correct_date_3, correct_type_3 = 0, 0, 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels_style = labels["style"].to(device).long()
            labels_date = labels["date"].to(device).long()
            labels_type = labels["type"].to(device).long()

            # Forward pass
            style_out, date_out, type_out, emb = model(images)

            # Predictions
            _, predicted_style_1 = torch.max(
                style_out, 1
            )  # Top-1 predictions for style
            _, predicted_date_1 = torch.max(date_out, 1)  # Top-1 predictions for date
            _, predicted_type_1 = torch.max(type_out, 1)  # Top-1 predictions for type

            # Top-3 predictions
            _, predicted_style_3 = torch.topk(style_out, 3, dim=1)
            _, predicted_date_3 = torch.topk(date_out, 3, dim=1)
            _, predicted_type_3 = torch.topk(type_out, 3, dim=1)

            # Increment correct counts for Top-1
            correct_style_1 += (predicted_style_1 == labels_style).sum().item()
            correct_date_1 += (predicted_date_1 == labels_date).sum().item()
            correct_type_1 += (predicted_type_1 == labels_type).sum().item()

            # Increment correct counts for Top-3
            correct_style_3 += sum(
                [
                    labels_style[i] in predicted_style_3[i]
                    for i in range(labels_style.size(0))
                ]
            )
            correct_date_3 += sum(
                [
                    labels_date[i] in predicted_date_3[i]
                    for i in range(labels_date.size(0))
                ]
            )
            correct_type_3 += sum(
                [
                    labels_type[i] in predicted_type_3[i]
                    for i in range(labels_type.size(0))
                ]
            )

            total += labels_style.size(0)

    # Compute accuracies
    style_accuracy_1 = 100 * correct_style_1 / total
    date_accuracy_1 = 100 * correct_date_1 / total
    type_accuracy_1 = 100 * correct_type_1 / total

    style_accuracy_3 = 100 * correct_style_3 / total
    date_accuracy_3 = 100 * correct_date_3 / total
    type_accuracy_3 = 100 * correct_type_3 / total

    print(f"Style Accuracy @1: {style_accuracy_1:.2f}%")
    print(f"Style Accuracy @3: {style_accuracy_3:.2f}%")
    print(f"Date Accuracy @1: {date_accuracy_1:.2f}%")
    print(f"Date Accuracy @3: {date_accuracy_3:.2f}%")
    print(f"Type Accuracy @1: {type_accuracy_1:.2f}%")
    print(f"Type Accuracy @3: {type_accuracy_3:.2f}%")
    return (
        style_accuracy_1,
        date_accuracy_1,
        type_accuracy_1,
        style_accuracy_3,
        date_accuracy_3,
        type_accuracy_3,
    )


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_style, num_classes_date, num_classes_type):
        super(MultiTaskModel, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)

        self.feature_extractor = backbone.features  # Extract features from MobileNetV3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Add pooling layer
        feature_dim = backbone.classifier[0].in_features  # Dimension of feature output

        # Task-specific heads
        self.fc_style = nn.Linear(feature_dim, num_classes_style)  # Style head
        self.fc_date = nn.Linear(feature_dim, num_classes_date)  # date head
        self.fc_type = nn.Linear(feature_dim, num_classes_type)  # Type head

        self.fc_first_emb = nn.Linear(feature_dim, 256)  # Embedding layer
        self.hw1 = nn.Hardswish()
        self.fc_emb = nn.Linear(256, 128)  # Embedding layer

    def forward(self, x):
        features = self.feature_extractor(x)  # Extract features
        features = self.avgpool(features)  # Pool to (batch_size, feature_dim, 1, 1)
        features = features.view(
            features.size(0), -1
        )  # Flatten to (batch_size, feature_dim)

        # Task-specific outputs
        style_out = self.fc_style(features)
        date_out = self.fc_date(features)
        type_out = self.fc_type(features)

        emb_out = self.fc_first_emb(features)
        emb_out = self.hw1(emb_out)
        emb_out = self.fc_emb(emb_out)

        return style_out, date_out, type_out, features
