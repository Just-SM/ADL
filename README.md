# Phase 2 Results

## Progress :

- Data Scraping ✅
- Images Scraping ✅
- Data Preprocessing ✅
- Pipeline base model ✅
- Basic embedding modeling ✅
- Base tests and Approximation algorithm for search ✅
- Full model train and test ✅
- UI application ✅

## Added:
- CI integration with pre-hooks ✅
- Docker compose build ✅

## Major structure changes:

- Clarified project structure.
- Migrated to script base development. (With exception of data_mining)
- Dataloader overhaul. Now with plane CSV.
- Moved from author school of art to time period painting feature. (The author school data is extremely hard to predict )

## Model structure:

Base: mobilenet_v3_small from pytorch

Adapted with : Multi Parameter Clarification task with painting data.

~~Fine tuned: with Contrastive learning for vector embedding.~~ _Showed to be ineffective. More details in report._

## New structure:
```
|
+-- data    # Contains Image data, model parameters and Image embeddings
|   |- art_data_loaded.csv  # Images metadata
|   |- embeddings.pth   # Precomputed Images embeddings
|   |- model.pth    # Model weights
|
+-- data_mining    # Images and images metadata scraping notebooks
|   |- DataScraping.ipynb   # Collect images metadata
|   |- ImagesScraping.ipynb   # Download Images
|
+-- dataloaders   # Torch Dataset and Dataloader provider
|   |- PaintingDatasets.py   # Torch Dataset and Dataloader
|
+-- models   # Torch model
|   |- MulitTaskModel.py   # Model and Evaluation
|
+-- notebooks   # Old notebooks for embeddings
|   |- Show_Embedings.ipynb   # Umap projection plot
|
+-- scripts   # Runner scripts for model training and embedding creation
|   |- generate_embedding.py   # Embeddings generation
|   |- train.py   # Train loop
|
+-- test   # Unit tests for model and Dataset/Dataloader
|   |- test_model.py   # Model tests
|   |- test_paintings_dataset.py   # Dataset and Dataloader tests
|
+-- streamlit_app.py  # Web application runner 
+-- requirements.txt  # Requirements for the project

```

## Metrics and results:

Planned:

🎯 Style/Date/Type Precision: >= 0.75

Achieved:

- Style Accuracy @1: 37.77%
- Style Accuracy @3: 78.25%✅
- Date Accuracy @1: 56.57%
- Date Accuracy @3: 93.15%✅
- Type Accuracy @1: 66.33%
- Type Accuracy @3: 90.30%✅


- Validation Top-1 Accuracy for Style: 37.77%
- Validation Top-1 Accuracy for Date: 56.57%
- Validation Top-1 Accuracy for Type: 66.33%

## To reproduce:

In order ⬇️

- DataScraping.ipynb (Put results in _data_)
- ImagesScraping.ipynb (Put results in _data_)
- `py train.py `
- `py generate_embedding.py`

# ~~Phase 1 Results~~ **_Outdated_**

## Plan of Work 📅

- Additional research on the topic (~4 hours)
- Data collection scripts and verification  (~5 hours)
- Small model creation (~3? hours)
- Small model testing (~30 min)
- Pipeline creation (~1 hours)
- Model training ( ?0 hours )
- Proof of concept testing (~30 min)
- Web dashboard interface (~4 hours)

Progress :

- Data Scraping ✅
- Images Scraping ✅
- Data Preprocessing ✅
- Train Indexes Pre-computation ✅
- Pipeline base model ✅
- Basic embedding modeling ✅
- Base tests and Approximation algorithms for search ❌
- Full model train and test ❌
- UI application ❌

## Details 

>❗
> THE **FULL INDEXES** files ARE **NOT UPDATED** (GIT  size limitation).


### Data Scraping

Deliverables:

>/data (Retrieved data)

>/data_mining (Notebooks)

As mentioned in task description, the data is scraped from [Web Gallery of Art](https://www.wga.hu/index.html)

DataScraping.ipynb Is collecting the Artist (style, name, period ... ) data and combines it with art pieces data(img_catalog.txt) collected from website. Additionally basic data transformation is done.

### Data Scraping

Deliverables:

>/image_data (Retrieved data)

>/data_mining (Notebooks)

DataScraping.ipynb Is downloading images. Broken links are removed.


### Train Indexes Pre-computation

Deliverables:

>/indexes_data (Retrieved data)

>/Create_full_test_set.ipynb
>/Create_mini_test_set.ipynb

Full data set considers:

Each image has school type and style, as characteristics. For training purposes each image (anchor) contains a set of all Ideal positive examples and all set of Ideal negative examples. Ideal means (**full** characteristics match or not match). 

For efficiency the data split into groups(f.e. French - Impressionism - Landscape ) as all of this group will have the same negative and positive examples.

The mini data (used for evaluation and as a test) is considers only one characteristic (style (as the most difficult)). Everything is the same.


### Pipeline base model

>/model (Trained model)

>/BaseLine_pipe.ipynb 


The basic model is can be represented as follows:

![piplene](readme_images\pipline.png)

Where the x1 and x2 are images. (Anchor and positive or negative example)

f0 is the CNN model.

Features are Fully connected layers, that are being trained.

Comparison is Contrastive loss function.

![piplene](readme_images\formula.png)


For the data, custom dataset and dataloader are created.

For the base model, the pre-trained 

>ShuffleNet_V2_X1

was used as lightweight CNN layers for feature extraction.

### Model modification

For the training purposed the CNN layers are frozen and the last clarification layers are replaced with dense layers.


### Basic embedding modeling

>/Show_Embeddings.ipynb

>/space.png

The notebook is using the trained baseline model for creating basic embeddings for small subset of mini_test_set.

The embedding are later projected to 2d space with UMAP and displayed on graph.


![space](readme_images\space.png)


One can observe that Model is creating some more-less meaningful clusters.  
