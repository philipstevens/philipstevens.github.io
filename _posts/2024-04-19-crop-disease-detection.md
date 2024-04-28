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

Detecting crop diseases early is crucial for food security and getting the most out of our harvests. My goal is to create a model that can spot these diseases accurately just by looking at images of leaves.

To do this, I'm turning to the [PlantVillage](https://www.plantvillage.org) dataset. It's got over 54,000 images showing both healthy and sick leaves, sorted into 38 groups based on what kind of plant it is and what disease it might have. You can find this dataset in the [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/plant_village).

I'm going to try two different approaches using Keras: first, I'll build a custom Convolutional Neural Network from scratch, and then I'll fine-tune an EfficientNet model.

# Setup

```python
!pip install -q --upgrade tensorflow
!pip install -q --upgrade keras
```

```python
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from keras import layers
from keras.applications import EfficientNetB0
from sklearn.metrics import classification_report
```

# Data

I'll split the data into three parts: 80% for training, 10% for validation, and 10% for testing. This ensures I've got solid sets for each step. I'll batch and shuffle the data, getting it ready for efficient processing and stopping the model from memorizing the order of samples. During training, the dataset will prefetch by default, keeping the flow smooth and minimizing downtime to make the most of the model training runs.

On top of that, I'll pull out the class names for easy reference later. The labels will be encoded as integers in the same order as the list, making it simple to connect with other parts of my setup.


```python
BATCH_SIZE = 128

(train, val, test), metadata = tfds.load(
    'plant_village',
    split=['train[:80%]','train[80%:90%]','train[90%:]'],
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

NUM_CLASSES = metadata.features['label'].num_classes
CLASS_NAMES = metadata.features['label'].names
```

```python
plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8")) #Read from dict
        plt.title(CLASS_NAMES[int(labels[i])], fontsize=10) #Lower font size
        plt.axis("off")
```

![png](Crop%20Disease%20Detection_files/Crop%20Disease%20Detection_9_0.png)
    

# Custom Model



## Build
To start, I’ll craft a custom CNN model using the Keras Functional API. Here's a detailed breakdown of the architecture:

1. **Input and Preprocessing**: It takes the images as input, which are expected to have dimensions of 256x256 pixels and 3 color channels (RGB). The raw images from the dataset come like this. The input images then undergo some data augmentation, which includes transformations like rotation, flipping, translation and adjustments to contrast to increase the diversity of the training data. I also normalize the images so that the pixel values of the images are scaled down to a range between 0 and 1.
2. **Feature Extraction**: The initial convolutional layer operates on normalized images, employing 128 filters sized at 3x3 pixels with a stride of 2 pixels. Batch normalization and ReLU activation are applied here and after most convolutional layers that follow.
    
    Following this, there are three blocks each with three convolutional layers. The number of feature maps in each block doubles compared to the previous block. Within each block, two consecutive layers use separable convolution for speed and efficiency, followed by max-pooling to reduce the feature map size by half. Simultaneously, there's another convolutional layer running in parallel. Its output connects with a residual connection, which adds its result to the output of the max-pooling layer. This combined output is then forwarded to the next block.
    
    One final convolutional layer is added to the output of the three blocks concluding in 1024 16x16 feature maps.
    
3. **Output**: After the feature extraction, global average pooling is applied to reduce the spatial dimensions of the feature maps to a single vector with 1024 dimensions for each image. Dropout is applied to the pooled features to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training. Finally, a dense layer produces the output logits, which represent the unnormalized scores for each of the 38 classes. No activation function is applied to this layer, as it aims to return raw logit scores rather than probabilities.


```python
data_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def build_custom_model(num_classes):
    # Input
    inputs = keras.Input(shape=(256, 256,3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # Feature Extraction
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)

    return keras.Model(inputs, outputs, name="CustomModel")


custom_model = build_custom_model(num_classes=NUM_CLASSES)
keras.utils.plot_model(custom_model, show_shapes=True)
```




    
![png](Crop%20Disease%20Detection_files/Crop%20Disease%20Detection_12_0.png)
    



## Train

In this phase, I train the custom model employing the Adam optimizer with a learning rate set to 0.0003. For training, I employ sparse categorical cross-entropy loss and track the accuracy metric. Throughout training, I save model checkpoints at the conclusion of each epoch. The training spans 25 epochs utilizing the training data, with validation data employed for validation throughout. The training history is then returned, serving as the basis for plotting the training run.


```python
callbacks = [
    keras.callbacks.ModelCheckpoint("custom_model_{epoch:02d}-{val_loss:.2f}.keras"),
]
custom_model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
hist = custom_model.fit(
    train,
    epochs=25,
    callbacks=callbacks,
    validation_data=val,
)
```

## Evaluate
In this section, I visualize the model training and examine the classification report from sklearn, featuring metrics like precision, recall, F1-score, and support for each class. It also includes overall average and weighted average metrics for the per-class statistics.

Upon review, a couple of observations stand out:

1. The training seems to converge rapidly, yet there's significant jitteriness in the validation curve likely due to small batch size and high learning rate.
2. Overall accuracy on the test set looks solid at 93%, though we observe that certain classes with lower support exhibit less impressive precision or recall.

```python
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def print_report(model, data):
  pred = np.argmax(model.predict(data), axis=-1)
  y = np.concatenate([y for x, y in data], axis=0)
  report = classification_report(y, pred, target_names = CLASS_NAMES)
  print(report)
```

```python
plot_hist(hist)
```
    
![png](Crop%20Disease%20Detection_files/Crop%20Disease%20Detection_19_0.png)

    
```python
print_report(custom_model, test)
```

                                                   precision    recall  f1-score   support
    
                               Apple___Apple_scab       1.00      0.88      0.94        58
                                Apple___Black_rot       1.00      0.91      0.95        65
                         Apple___Cedar_apple_rust       1.00      1.00      1.00        25
                                  Apple___healthy       1.00      0.84      0.91       159
                              Blueberry___healthy       1.00      0.99      1.00       133
                                 Cherry___healthy       1.00      0.32      0.49        90
                          Cherry___Powdery_mildew       0.99      0.97      0.98       100
       Corn___Cercospora_leaf_spot Gray_leaf_spot       0.87      0.96      0.92        56
                               Corn___Common_rust       1.00      1.00      1.00       127
                                   Corn___healthy       1.00      1.00      1.00       112
                      Corn___Northern_Leaf_Blight       0.96      0.94      0.95       112
                                Grape___Black_rot       0.97      0.96      0.96       112
                     Grape___Esca_(Black_Measles)       1.00      0.96      0.98       126
                                  Grape___healthy       0.97      1.00      0.99        36
       Grape___Leaf_blight_(Isariopsis_Leaf_Spot)       1.00      0.83      0.91       105
         Orange___Haunglongbing_(Citrus_greening)       1.00      1.00      1.00       561
                           Peach___Bacterial_spot       0.91      0.98      0.95       237
                                  Peach___healthy       0.45      1.00      0.62        27
                    Pepper,_bell___Bacterial_spot       0.97      0.98      0.98       105
                           Pepper,_bell___healthy       0.76      0.99      0.86       154
                            Potato___Early_blight       0.70      1.00      0.82        98
                                 Potato___healthy       0.89      0.80      0.84        10
                             Potato___Late_blight       0.88      0.95      0.92       104
                              Raspberry___healthy       1.00      0.90      0.95        30
                                Soybean___healthy       0.95      1.00      0.97       527
                          Squash___Powdery_mildew       0.92      1.00      0.96       188
                             Strawberry___healthy       1.00      0.76      0.86        50
                         Strawberry___Leaf_scorch       1.00      0.90      0.95        88
                          Tomato___Bacterial_spot       0.95      0.97      0.96       184
                            Tomato___Early_blight       0.74      0.97      0.84        96
                                 Tomato___healthy       0.98      1.00      0.99       164
                             Tomato___Late_blight       0.99      0.75      0.86       191
                               Tomato___Leaf_Mold       1.00      0.57      0.72        97
                      Tomato___Septoria_leaf_spot       0.95      0.85      0.89       181
    Tomato___Spider_mites Two-spotted_spider_mite       0.72      0.99      0.83       168
                             Tomato___Target_Spot       0.90      0.94      0.92       149
                     Tomato___Tomato_mosaic_virus       0.97      0.97      0.97        36
           Tomato___Tomato_Yellow_Leaf_Curl_Virus       1.00      0.93      0.96       569
    
                                         accuracy                           0.93      5430
                                        macro avg       0.93      0.91      0.91      5430
                                     weighted avg       0.95      0.93      0.93      5430
    


# Fine-tune EfficientNet

Rather than using a custom model, let's finetune a pretrained EfficientNet model.

## Build
Here I construct my model by harnessing the power of the EfficientNet architecture along with pre-trained weights. Here's a detailed breakdown of the process:

1. **Input and Preprocessing**:  The variant of EfficientNet I'm utilizing expects images with dimensions of 224x224 pixels. Therefore, I ensure that the images are resized accordingly to meet this requirement. The input also undergoes the same data augmentation as the custom model did. No normalizing is done here as the EfficientNet already handles all other aspects of preprocessing.
2. **EfficientNetB0:** I employ the EfficientNetB0 model that's been pre-trained on the ImageNet dataset. Since we won't be using the top layer for ImageNet classes, I discard it. Additionally, I freeze the weights to preserve the learned representations, keeping them static during the initial training phases.
3. **Output**: To tailor the model for our specific classification task, I introduce additional layers atop the pre-trained base. This includes incorporating global average pooling, batch normalization, dropout, and a dense layer devoid of an activation function, which yields the class predictions.



```python
def build_efficientnet_model(num_classes):

    # Input
    inputs = layers.Input(shape=(256, 256, 3))
    x = data_augmentation(inputs)
    x = layers.Resizing(224, 224)(x)

    # Pretrained model with frozen weights and top layer removed
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
    model.trainable = False

    # Output
    x = layers.GlobalAveragePooling2D()(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)

    return keras.Model(inputs, outputs, name="EfficientNet")

en_model = build_efficientnet_model(num_classes=NUM_CLASSES)
keras.utils.plot_model(en_model)
```

## Train
First, I'll train the custom top layer of this model using a relatively high learning rate to quickly get a decent model. Then, I'll unfreeze the EfficientNetB0 layers and train again with a smaller learning rate. This method allows for only subtle adjustments to be made to the pre-trained weights so we don't lose any critical feature representations in the initial stages of training. The idea is to keep the underlying power of the pretrained model and make it work better for our specific needs.

### Step 1


```python
callbacks = [
    keras.callbacks.ModelCheckpoint("efficientnet_model_step_1_{epoch:02d}-{val_loss:.2f}.keras"),
]

en_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

step1_hist = en_model.fit(
    train,
    epochs=25,
    callbacks=callbacks,
    validation_data=val,
)
```

### Step 2


```python
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

unfreeze_model(en_model)
```


```python
callbacks = [
    keras.callbacks.ModelCheckpoint("efficientnet_model_step_2_{epoch:02d}-{val_loss:.2f}.keras"),
]

en_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)

step2_hist = en_model.fit(
    train,
    epochs=4,
    callbacks=callbacks,
    validation_data=val)
```

## Evaluate
As before, I visualize the model training and examine the per-class and overall classification metrics.

Here, a few observations stand out:

1. The training run reaches high accuracy even before I unfreeze the bottom pre-trained layers. Again there is some jitteriness in the validation curve in stage 1 likely due to small batch size and high learning rate. However, further training these layers results in a slight enhancement in overall accuracy and a more robust model.
2. The validation accuracy is higher than the training accuracy. This is common behaviour when using dropout layers, as they are only used in training and deliberately hamper the model.
2. Overall accuracy on the test set impressively reaches 97%, with all classes exhibiting high precision and recall. Nonetheless, some classes still demonstrate slightly better performance than others.








```python
plot_hist(step1_hist)
```


    
![png](Crop%20Disease%20Detection_files/Crop%20Disease%20Detection_32_0.png)
    



```python
plot_hist(step2_hist)
```


    
![png](Crop%20Disease%20Detection_files/Crop%20Disease%20Detection_33_0.png)
    



```python
print_report(en_model, test)
```

                                                   precision    recall  f1-score   support
    
                               Apple___Apple_scab       0.96      0.93      0.95        58
                                Apple___Black_rot       1.00      1.00      1.00        65
                         Apple___Cedar_apple_rust       1.00      0.88      0.94        25
                                  Apple___healthy       0.98      0.99      0.99       159
                              Blueberry___healthy       0.99      1.00      1.00       133
                                 Cherry___healthy       1.00      0.99      0.99        90
                          Cherry___Powdery_mildew       0.99      1.00      1.00       100
       Corn___Cercospora_leaf_spot Gray_leaf_spot       0.92      0.88      0.90        56
                               Corn___Common_rust       0.98      1.00      0.99       127
                                   Corn___healthy       1.00      0.99      1.00       112
                      Corn___Northern_Leaf_Blight       0.94      0.96      0.95       112
                                Grape___Black_rot       0.96      0.96      0.96       112
                     Grape___Esca_(Black_Measles)       0.98      0.97      0.97       126
                                  Grape___healthy       1.00      1.00      1.00        36
       Grape___Leaf_blight_(Isariopsis_Leaf_Spot)       1.00      1.00      1.00       105
         Orange___Haunglongbing_(Citrus_greening)       1.00      1.00      1.00       561
                           Peach___Bacterial_spot       1.00      0.99      0.99       237
                                  Peach___healthy       0.93      1.00      0.96        27
                    Pepper,_bell___Bacterial_spot       0.99      0.98      0.99       105
                           Pepper,_bell___healthy       0.99      1.00      0.99       154
                            Potato___Early_blight       0.96      0.99      0.97        98
                                 Potato___healthy       1.00      0.80      0.89        10
                             Potato___Late_blight       0.97      0.96      0.97       104
                              Raspberry___healthy       1.00      1.00      1.00        30
                                Soybean___healthy       0.99      1.00      1.00       527
                          Squash___Powdery_mildew       1.00      1.00      1.00       188
                             Strawberry___healthy       0.98      1.00      0.99        50
                         Strawberry___Leaf_scorch       1.00      0.98      0.99        88
                          Tomato___Bacterial_spot       0.94      0.93      0.94       184
                            Tomato___Early_blight       0.86      0.66      0.75        96
                                 Tomato___healthy       0.96      0.99      0.98       164
                             Tomato___Late_blight       0.87      0.95      0.91       191
                               Tomato___Leaf_Mold       0.96      0.84      0.90        97
                      Tomato___Septoria_leaf_spot       0.94      0.88      0.91       181
    Tomato___Spider_mites Two-spotted_spider_mite       0.90      0.96      0.93       168
                             Tomato___Target_Spot       0.83      0.95      0.89       149
                     Tomato___Tomato_mosaic_virus       0.97      0.97      0.97        36
           Tomato___Tomato_Yellow_Leaf_Curl_Virus       0.99      0.98      0.98       569
    
                                         accuracy                           0.97      5430
                                        macro avg       0.97      0.96      0.96      5430
                                     weighted avg       0.97      0.97      0.97      5430
    



