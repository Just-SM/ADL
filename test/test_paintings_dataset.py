import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
import torch
from PIL import Image
import pandas as pd 
from torch.utils.data import DataLoader
from torchvision.transforms import v2


from dataloaders.PaintingDatasets import PaintingsDataset, compute_weights, CreatePaintingsDataLoaders


@pytest.fixture
def sample_csv_with_images(tmp_path):
    """Fixture to create a temporary CSV file and dummy images for testing."""
    # Create a directory for images
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Generate dummy images
    for i in range(3):
        img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))  # Dummy RGB image
        img.save(image_dir / f"{i}.jpg",)

    # Create a CSV file referencing these images
    sample_data = pd.DataFrame({
        "image_path": [str(image_dir / f"{i}") for i in range(3)],
        "style": ["Impressionism", "Baroque", "Modernism"],
        "date": ["1875", "1600", "1920"],
        "type": ["landscape", "religious", "landscape"],
    })
    csv_file = tmp_path / "test_paintings.csv"
    sample_data.to_csv(csv_file, index=False)

    return (str(csv_file),str(image_dir))


def test_dataset_length(sample_csv_with_images):
    """Test the length of the dataset."""

    csv_file,image_dir = sample_csv_with_images
    dataset = PaintingsDataset(csv_file=csv_file,data_folder=image_dir)
    assert len(dataset) == 3, "Dataset length should match the number of rows in the CSV."


def test_dataset_getitem(sample_csv_with_images):
    """Test the __getitem__ method of the dataset."""
    csv_file,image_dir = sample_csv_with_images
    dataset = PaintingsDataset(csv_file=csv_file,data_folder=image_dir)
    image, labels = dataset[0]
    assert isinstance(labels, dict), "Labels should be a dictionary."
    assert "style" in labels, "Labels should include 'style'."
    assert "date" in labels, "Labels should include 'date'."
    assert "type" in labels, "Labels should include 'type'."


def test_transform(sample_csv_with_images):
    """Test if transformations are applied correctly."""
    csv_file,image_dir = sample_csv_with_images
    transform = v2.Compose([v2.Resize((224, 224))])
    dataset = PaintingsDataset(csv_file=csv_file,data_folder=image_dir, transform=transform)
    image, labels = dataset[0]
    assert image.shape[-2:] == (224, 224), "Image should be resized to 224x224."


def test_dataloaders(sample_csv_with_images):
    """Test if CreatePaintingsDataLoaders returns valid dataloaders and weights."""
    csv_file,image_dir = sample_csv_with_images
    batch_size = 2
    train_loader, val_loader, weights, unique_classes = CreatePaintingsDataLoaders(csv_file=csv_file,data_folder=image_dir, batch_size=batch_size)

    # Check if loaders are DataLoader instances
    assert isinstance(train_loader, DataLoader), "Train loader should be a DataLoader."
    assert isinstance(val_loader, DataLoader), "Validation loader should be a DataLoader."

    # Check weights
    assert len(weights) == 3, "Weights list should have 3 elements for 'style', 'date', and 'type'."

    # Check unique classes
    assert len(unique_classes) == 3, "Unique classes should have 3 elements for 'style', 'date', and 'type'."
    assert "Impressionism" in unique_classes[0], "Unique classes for 'style' should include 'Impressionism'."
    assert '1550 - 1650' in unique_classes[1], "Unique classes for 'date' should include '1550 - 1650'."
    assert "landscape" in unique_classes[2], "Unique classes for 'type' should include 'landscape'."


def test_compute_weights(sample_csv_with_images):
    csv_file,image_dir = sample_csv_with_images
    """Test the compute_weights function."""
    dataset = PaintingsDataset(csv_file=csv_file,data_folder=image_dir)
    dataset.data["style_encoded"] = [0, 1, 0]
    style_weights = compute_weights(dataset, "style_encoded")
    assert len(style_weights) == 2, "Style weights should have the same number of unique classes."
    assert torch.is_tensor(style_weights), "Style weights should be a PyTorch tensor."
