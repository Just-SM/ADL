import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from models.MulitTaskModel import MultiTaskModel, evaluate



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyDataset(Dataset):
    """A dummy dataset for testing purposes."""
    def __init__(self, num_samples, num_classes_style, num_classes_date, num_classes_type):
        self.num_samples = num_samples
        self.num_classes_style = num_classes_style
        self.num_classes_date = num_classes_date
        self.num_classes_type = num_classes_type

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a dummy image and dummy labels
        image = torch.rand(3, 224, 224)  # Random image with 3 channels and size 224x224
        labels = {
            "style": torch.randint(0, self.num_classes_style, (1,)).item(),
            "date": torch.randint(0, self.num_classes_date, (1,)).item(),
            "type": torch.randint(0, self.num_classes_type, (1,)).item(),
        }
        return image, labels


@pytest.fixture
def dummy_dataloader():
    """Fixture to create a dummy DataLoader."""
    num_samples = 10
    num_classes_style = 5
    num_classes_date = 4
    num_classes_type = 3

    dataset = DummyDataset(num_samples, num_classes_style, num_classes_date, num_classes_type)
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader, num_classes_style, num_classes_date, num_classes_type


@pytest.fixture
def multitask_model(dummy_dataloader):
    """Fixture to create a MultiTaskModel instance."""
    _, num_classes_style, num_classes_date, num_classes_type = dummy_dataloader
    model = MultiTaskModel(
        num_classes_style=num_classes_style,
        num_classes_date=num_classes_date,
        num_classes_type=num_classes_type,
    ).to(device)
    return model


def test_model_forward_pass(multitask_model):
    """Test the forward pass of the MultiTaskModel."""
    model = multitask_model
    batch_size = 4
    dummy_input = torch.rand(batch_size, 3, 224, 224).to(device)
    
    style_out, school_out, type_out, embeddings = model(dummy_input)

    # Check output dimensions
    assert style_out.shape == (batch_size, model.fc_style.out_features), "Style output shape mismatch."
    assert school_out.shape == (batch_size, model.fc_date.out_features), "Date output shape mismatch."
    assert type_out.shape == (batch_size, model.fc_type.out_features), "Type output shape mismatch."
    assert embeddings.shape == (batch_size, model.fc_first_emb.in_features), "Embedding output shape mismatch."


def test_evaluate_function(multitask_model, dummy_dataloader):
    """Test the evaluate function with the model and dummy DataLoader."""
    model = multitask_model
    dataloader, _, _, _ = dummy_dataloader

    # Perform evaluation
    results = evaluate(model, dataloader)

    # Ensure results are returned for all accuracy metrics
    assert len(results) == 6, "Evaluate function should return 6 accuracy values."
    assert all(isinstance(acc, float) for acc in results), "All returned accuracy values should be floats."
    assert all(0 <= acc <= 100 for acc in results), "Accuracy values should be between 0 and 100."
