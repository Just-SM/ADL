import torch
from torchvision.io import decode_image
from tqdm import tqdm
import pandas as pd
import sys
import os


# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.MulitTaskModel import MultiTaskModel
from dataloaders.PaintingDatasets import DEFAULT_TRANSFORMER_PIPE, bin_dates


device = "cuda" if torch.cuda.is_available() else "cpu"
device


DATA_FOLDER = "data"
IMAGE_RAW_DATA = "raw_images"
DATA_CSV = "art_data_loaded.csv"


df = pd.read_csv(
    os.path.join(
        DATA_FOLDER,
        DATA_CSV,
    )
)

style_size = len(df["style"].unique())
style_date = len(df["date"].apply(bin_dates).unique())
style_type = len(df["type"].unique())
model = MultiTaskModel(style_size, style_date, style_type)


model.load_state_dict(
    torch.load(
        os.path.join(
            DATA_FOLDER,
            "model.pth",
        ),
        weights_only=True,
    )
)
model.to(device)


val_dict = dict()

for id in tqdm(df["index"]):
    with torch.no_grad():
        img = decode_image(
            os.path.join(
                os.path.join(
                    DATA_FOLDER,
                    IMAGE_RAW_DATA,
                ),
                f"{id}.jpg",
            )
        )

        # Step 3: Apply inference preprocessing transforms
        batch = DEFAULT_TRANSFORMER_PIPE(img).unsqueeze(0)

        batch = batch.to(device)

        style_out, date_out, type_out, emb_out = model(batch)

        emb_out = emb_out.to(device)

        val_dict[id] = emb_out


torch.save(
    val_dict,
    os.path.join(
        DATA_FOLDER,
        "embeddings.pth",
    ),
)
