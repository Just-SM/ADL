# ImageToVec encoder for proximity search.

## Reference papers 📃:
- [SimCLR. A Simple Framework for Contrastive Learning of Visual Representations.](https://arxiv.org/abs/2002.05709)
- [Clustering-based Contrastive Learning for Improving Face Representations](https://arxiv.org/pdf/2004.02195)
- [Locality-Sensitive Hashing for Finding Nearest Neighbors](https://ieeexplore.ieee.org/abstract/document/4472264)
- ...

## Topic 

### *Computer Vision* 🤖

Image encoding to vector space.


## Project type 

### *Hybrid*
Combination of **Bring your data** and **Bring your method**.

## Project description

### Idea 🖼️
The core idea is to encode the image to a vector space for future proximity search and clustering.

In particular, I am interested in painting. Imagine painting something and then finding out which style, years, school, or authors it is most similar to.

### Model and Techniques 🔧
The current plan is to train/fine-tune CNN with Contrast learning techniques to create an Image encoder. Later, some locality approximation algorithms will be used to optimize search and clustering.

### Data 💾
Data will be collected (scrapped) from the [Web Gallery of Art](https://www.wga.hu/index.html), which has approximately 50,000 art pieces and 6,000 authors. Of course, some additional sources might be considered.

## Plan of Work 📅

- Additional research on the topic (~4 hours)
- Data collection scripts and verification  (~5 hours)
- Small model creation (~3? hours)
- Small model testing (~30 min)
- Piplene creation (~1 hours)
- Model training ( ?0 hours )
- Proof of concept testing (~30 min)
- Web dashboard interface (~4 hours)
