---
title: "Crop Disease Detection in Images"
date: 2024-04-19
categories:
  - computer vision
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
---
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1w4vPmpTiKWLiIjrrkWBJ-Ry-VP9XCqns?usp=sharing)

In modern agriculture, early detection of crop diseases is paramount for ensuring food security and maximizing yields. My aim here is to develop a model capable of accurately identifying crop diseases from images of leaves.

I will use the PlantVillage dataset. This dataset comprises 54,303 images of both healthy and unhealthy leaves, categorized into 38 groups based on species and disease. This dataset is accessible via TensorFlow Datasets.

I will explore two different approaches using Keras: a hand-built custom Convolutional Neural Network and a fine-tuned EfficientNet model.

