import streamlit as st
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
import os


# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


from models.MulitTaskModel import MultiTaskModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on : ", device)


st.set_page_config(
    page_title="PaintApp",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def prep_images_emb():
    print("Loading embeddings")

    dict_data = torch.load(os.path.join("data", "embeddings.pth"))

    indexes = []

    emb = []

    for k, v in dict_data.items():
        indexes.append(k)
        emb.append(v[0])

    print("Loaded")
    return indexes, emb


def prep_images_nn():
    print("Fitting NNB")

    return NearestNeighbors(metric="cosine", algorithm="brute").fit(
        st.session_state["images_emb"]
    )


def prep_images_data():
    df = pd.read_csv(os.path.join("data", "art_data_loaded.csv"))

    return df


def color_precent_text(val):
    if val > 6:
        return f":green[{val}0 %]"
    elif val > 4:
        return f":orange[{val}0 %]"
    else:
        return f":red[{val}0 %]"


def check_for_exact_match(sim):
    return sim[0] < 0.18 and abs(sim[0] - sim[1]) > sum(
        [abs(x - y) for x, y in zip(sim[1:], sim[2:])]
    ) / len(sim - 2)


def bin_dates(date):
    if date < 1350:
        return "pre 1350"
    elif date < 1450:
        return "1350 - 1450"
    elif date < 1550:
        return "1450 - 1550"
    elif date < 1650:
        return "1550 - 1650"
    elif date < 1750:
        return "1650 - 1750"
    elif date < 1850:
        return "1750 - 1850"
    elif date < 1950:
        return "1850 - 1950"
    elif date < 2000:
        return "1950 - 2000"
    else:
        return "post 2000"


def load_model():
    style_size = len(st.session_state["images_data"]["style"].unique())
    style_date = len(st.session_state["images_data"]["date"].apply(bin_dates).unique())
    style_type = len(st.session_state["images_data"]["type"].unique())
    model = MultiTaskModel(style_size, style_date, style_type)

    # model = MultiTaskModel(12, 8, 10)
    model.load_state_dict(
        torch.load(os.path.join("data", "model.pth"), weights_only=True)
    )

    model = model.to(device)
    model.eval()

    return model


if "images_data" not in st.session_state:
    st.session_state["images_data"] = prep_images_data()


if "model" not in st.session_state:
    st.session_state["model"] = load_model()


if "transformer_pipe" not in st.session_state:
    st.session_state["transformer_pipe"] = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


if "images_emb" not in st.session_state:
    indexes, embeds = prep_images_emb()

    st.session_state["images_index"] = indexes
    st.session_state["images_emb"] = embeds

    st.session_state["nnb"] = prep_images_nn()


st.markdown("# Select an Image")


image = st.file_uploader(
    "Upload image",
    ["png", "jpg"],
    accept_multiple_files=False,
)


if image:
    image_tensor = decode_image(
        torch.tensor(bytearray(image.getvalue()), dtype=torch.uint8),
        mode=ImageReadMode.RGB,
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.markdown("### Selected image: ")
    with col2:
        st.image(image, width=300)

    st.divider()

    batch = st.session_state["transformer_pipe"](image_tensor).unsqueeze(0)

    batch = batch.to(device)

    style_out, date_out, type_out, emb_out = st.session_state["model"](batch)

    emb_out = emb_out.detach().to("cpu")
    style_out = style_out.detach().to("cpu")
    date_out = date_out.detach().to("cpu")
    type_out = type_out.detach().to("cpu")

    dist, ind = st.session_state["nnb"].kneighbors(emb_out, 10)

    if check_for_exact_match(dist[0]):
        with col3:
            st.markdown("### Close match found! 🎉")

            st.markdown(
                f" ## {st.session_state['images_data'].iloc[ind[0][0]]['title']} "
            )
            st.markdown(
                f" ### {st.session_state['images_data'].iloc[ind[0][0]]['author']} "
            )
            type_str = st.session_state["images_data"].iloc[ind[0][0]]["type"]

        with col4:
            st.image(st.session_state["images_data"].iloc[ind[0][0]]["url"], width=300)

    types = []
    style = []
    date = []

    from collections import Counter

    for i in ind[0]:
        types.append(st.session_state["images_data"].iloc[i]["type"])
        style.append(st.session_state["images_data"].iloc[i]["style"])
        date.append(st.session_state["images_data"].iloc[i]["date"])

    types = sorted(Counter(types).items(), key=lambda x: x[1], reverse=True)
    style = sorted(Counter(style).items(), key=lambda x: x[1], reverse=True)
    date = sorted(
        Counter([bin_dates(x) for x in date]).items(), key=lambda x: x[1], reverse=True
    )

    st.markdown("# Results:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("## Type 🖼️")
        st.markdown(f"### {color_precent_text(types[0][1])} -  {types[0][0].upper()} ")
        for t, hashs in zip(types[1:3], ["####", "#####"]):
            st.markdown(f"{hashs} {color_precent_text(t[1])} -  {t[0].upper()} ")
    with col2:
        st.markdown("## Style 🖌️")
        st.markdown(f"### {color_precent_text(style[0][1])}-  {style[0][0].upper()} ")
        for t, hashs in zip(style[1:3], ["####", "#####"]):
            st.markdown(f"{hashs} {color_precent_text(t[1])} -  {t[0].upper()} ")
    with col3:
        st.markdown("## Time period 📅")
        st.markdown(f"### {color_precent_text(date[0][1])} -  ({date[0][0].upper()}) ")
        for t, hashs in zip(date[1:3], ["####", "#####"]):
            st.markdown(f"{hashs} {color_precent_text(t[1])} -  ({t[0].upper()}) ")

    st.divider()
    st.markdown("# Similar paintings: ")
    st.write(" ")

    for i, val in zip(ind[0], dist[0]):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.image(st.session_state["images_data"].iloc[i]["url"], width=300)
        with c2:
            st.markdown(f" ## {st.session_state['images_data'].iloc[i]['title']} ")
            st.markdown(f" ### {st.session_state['images_data'].iloc[i]['author']} ")
            type_str = st.session_state["images_data"].iloc[i]["type"]
            st.markdown(f" #### Type: *{type_str[0].upper() + type_str[1:]}*")
            st.markdown(
                f" #### Date: *{int(st.session_state['images_data'].iloc[i]['date'])}* "
            )
            st.markdown(
                f" #### Style: *{st.session_state['images_data'].iloc[i]['style']}* "
            )
            # st.markdown(f" #### Similarity: *{val}* ")

        st.divider()
