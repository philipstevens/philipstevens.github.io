---
title: "Basic Transfer Learning"
date: 2019-12-10T10:47:44+07:00
categories:
  - computer vision
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
---

# Basic Transfer Learning

Here I will demonstrate how to do basic transfer learning with tensorflow and keras. Transfer learning is when you reuse the model weights trained for one task as the starting point for training a model on a different but related task. The value of this is to very quickly develop a model that converges fast in training and performs well in use. 

In this demonstration the bottom layer of our neural network will use a feature vector extracted from a much more complex convolutional neural network trained on a very broad set of images. We will transfer this to the smaller more specific domain of images of certain cat and dog breeds.  


## Imports 

We will make use of the following modules:

```python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub # connects to a library of reusable machine learning model components called modules: https://tfhub.dev/
import tensorflow_datasets as tfds # connects to a collection of datasets ready to use with tensorflow: https://www.tensorflow.org/datasets/catalog/overview. 
import matplotlib.pyplot as plt # to plot our learning curves
```


## Setup and validate the GPUs

Here we prepare the GPUs for use and confirm they are available to Tensorflow. Note memory growth must be set before GPUs have been initialized and that memory growth needs to be the same across GPUs. The output lets us know this step worked and how many GPUs are available. 


```python
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

```

    1 Physical GPUs, 1 Logical GPUs


## Load dataset

Here we load in The Oxford-IIIT Pet Dataset found here: https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet. 

The Oxford-IIIT Pet Dataset is an image dataset of 37 breeds of cats and dogs with around 200 images for each breed. The images vary in scale, pose and lighting. The dataset also consists of the corresponding label for each image and a pixel-wise mask for image segmentation. 

tfds.load is a convenience method that's the simplest way to build and load a tf.data.Dataset when it is registered in the Tensorflow datasets collection. When with_info=True a tuple is returned that contains both the dataset and a tfds.core.DatasetInfo object which contains useful metadata on the dataset. 

Each dataset comes in it's own format which you may have to explore. In this case our dataset is actually two tf.data.Dataset's specified in a dictionary as the train and test splits. Each split contains the images, labels, file names and segmentation masks. 

Note when we load we could set as_supervised to true and the dataset will have a 2-tuple structure of (image, label) and remove all the other data we aren't using. However in this case the info object does not include a feature dictionary from label to human-readable string so we must infer it ourselves from the dataset directly where a file_name is associated with each label.


```python
dataset, info = tfds.load('oxford_iiit_pet', with_info=True)

raw_train = dataset['train']
raw_test = dataset['test']
```

## Prepare input pipeline

Here we prepare our input pipeline for training. Most importantly we format our dataset as a 2-tuple structure of (image, label), resize the images so they all have the same dimensions, and then batch the (image, label) tuples into groups of 34. This is applied to both the train and the test splits.

For the training data we also augment our dataset by randomly flipping images left and right. This effectively increases the size of our training set. We also shuffle and repeat the input pipeline where the batches and the augmentations are randomly re-applied each iteration. 


```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = info.splits['train'].num_examples

train = raw_train.map(
    lambda datapoint: (datapoint['image'], datapoint['label'])
).map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
).map(
    lambda image, label: (tf.image.resize(image, IMG_SIZE), label)
).map(
    lambda image, label: (tf.image.random_flip_left_right(image), label)
).cache().shuffle(
    SHUFFLE_BUFFER_SIZE
).batch(
    BATCH_SIZE
).repeat()

test = raw_test.map(
    lambda datapoint: (datapoint['image'], datapoint['label'])
).map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
).map(
    lambda image, label: (tf.image.resize(image, IMG_SIZE), label)
).batch(
    BATCH_SIZE
)
```

## View and validate images in pipeline

It's good to sanity check our input pipeline to make sure the images and labels are what we expect. 

First we want to create a dictionary to map from labels to human-readable strings. We can infer this from the file names contained in the original raw dataset. Next we want to display a number of images along with their breed.

I'm not personally an expert on dog and cat breeds but notice a breed starting with a capital letter indicates the species is a cat and lowercase indicates the species is a dog. Combine that knowledge with the few cat and dog breeds I am familiar with and all seems well. 


```python
def generate_name_id_pairs(datapoint):
    name = datapoint['file_name']
    label = datapoint['label']
    return name, label

name_id_pairs = raw_train.map(generate_name_id_pairs)

feat_dict = {}
for name, label in name_id_pairs:
    feat_dict[label.numpy()] = ' '.join(name.numpy().decode("utf-8").split('_')[:-1])

def plotImages(batch):
    img, label = batch
    
    plt.figure(figsize=(20,20))
    
    for n in range(30):
        
        plt.subplot(6,5,n+1)
        plt.imshow(img[n])
        plt.title(feat_dict[label[n].numpy()])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

sample_train_batch = next(iter(train))
plotImages(sample_train_batch) 

```


![png](/assets/images/basic-transfer-learning/output_12_0.png)


## Prepare model

Here we actually prepare our model.

First we fetch the feature vector from tensorflow hub. There are many to choose from but here we grab a mobilnet_v2_140_224. It is extracted from a mobilenetV2 model trained on 224x224 ImageNet images with a depth multiplier of 1.4. The magnitude of the depth multiplier corresponds to the number of features in the convolutional layers with 1.4 being the highest available. We set the input shape to include size of our images and the three rgb channels. We also don't want to change any of the weights in our input layer during training so we set it to not be trainable. 

Note that the feature vector is extracted from a model trained on the ILSVRC2012 dataset which is a subset of 1000 classes from ImageNet. Of the 37 pets included in The Oxford-IIIT Pet Dataset dataset 23 of them are in the ILSVRC2012 dataset. That is 21 of the 25 dog breeds and 2 of the 12 cat breeds are in the training set for the feature vector albeit under different labels. This means we will be particularly interested in how this model adapts to the new classes included in the Oxford-IIIT Pet dataset but not the ILSVRC2012 dataset. Here's the list of such classes (remember cats start with a capital letter and dogs do not):

Sphynx,
British Shorthair,
Maine Coon,
Abyssinian,
Bengal,
Egyptian Mau,
Russian Blue,
Ragdoll,
Birman,
Bombay,
havanese,
leonberger,
american bulldog,
shiba inu

We also include a dropout layer to reduce overfitting and add our prediction layer to the top.

The output describes our final model architecture. 


```python
OUTPUT_CHANNELS = 3
IMG_SHAPE = (*IMG_SIZE, OUTPUT_CHANNELS)
NUM_CLASSES = info.features['label'].num_classes

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
base_model = hub.KerasLayer(feature_extractor_url, trainable=False, input_shape=IMG_SHAPE)

model = tf.keras.Sequential([
    base_model,
    keras.layers.Dropout(0.25),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    keras_layer (KerasLayer)     (None, 1792)              4363712   
    _________________________________________________________________
    dropout (Dropout)            (None, 1792)              0         
    _________________________________________________________________
    dense (Dense)                (None, 37)                66341     
    =================================================================
    Total params: 4,430,053
    Trainable params: 66,341
    Non-trainable params: 4,363,712
    _________________________________________________________________


## Prepare training

We use the Adam optimizer and set it's learning rate to 0.0005 which is half it's default value. Our labels are integers and categorical, so rather than one-hot encode them we can simply use the sparse categorical cross entropy loss. We will use accuracy as our metric to understand how well our models are performing.


```python
LEARNING_RATE = 0.001*0.5

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## Train model

We train for 30 epochs and since we set our training input pipeline to repeat we are required to explicitly state the number of steps per epoch. We also set the number of validation steps in such a way that we have 5 subsplits of our validation data. 


```python
EPOCHS = 30

TRAIN_LENGTH = info.splits['train'].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS


history = model.fit(train, 
                    validation_data=test,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS
                    )
```

    Train for 115 steps, validate for 22 steps
    Epoch 1/30
    115/115 [==============================] - 11s 95ms/step - loss: 1.6393 - accuracy: 0.6046 - val_loss: 0.6802 - val_accuracy: 0.8494
    Epoch 2/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.4478 - accuracy: 0.8924 - val_loss: 0.4426 - val_accuracy: 0.8892
    Epoch 3/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.2859 - accuracy: 0.9318 - val_loss: 0.3773 - val_accuracy: 0.8991
    Epoch 4/30
    115/115 [==============================] - 6s 54ms/step - loss: 0.2167 - accuracy: 0.9462 - val_loss: 0.3480 - val_accuracy: 0.8935
    Epoch 5/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.1736 - accuracy: 0.9579 - val_loss: 0.3222 - val_accuracy: 0.9162
    Epoch 6/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.1372 - accuracy: 0.9739 - val_loss: 0.3103 - val_accuracy: 0.9134
    Epoch 7/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.1155 - accuracy: 0.9772 - val_loss: 0.3014 - val_accuracy: 0.9148
    Epoch 8/30
    115/115 [==============================] - 6s 54ms/step - loss: 0.1034 - accuracy: 0.9804 - val_loss: 0.2868 - val_accuracy: 0.9105
    Epoch 9/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.0822 - accuracy: 0.9883 - val_loss: 0.2857 - val_accuracy: 0.9091
    Epoch 10/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0729 - accuracy: 0.9899 - val_loss: 0.2919 - val_accuracy: 0.9148
    Epoch 11/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0645 - accuracy: 0.9924 - val_loss: 0.2807 - val_accuracy: 0.9148
    Epoch 12/30
    115/115 [==============================] - 6s 51ms/step - loss: 0.0572 - accuracy: 0.9935 - val_loss: 0.2781 - val_accuracy: 0.9190
    Epoch 13/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0501 - accuracy: 0.9951 - val_loss: 0.2684 - val_accuracy: 0.9205
    Epoch 14/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.0448 - accuracy: 0.9962 - val_loss: 0.2768 - val_accuracy: 0.9190
    Epoch 15/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0402 - accuracy: 0.9973 - val_loss: 0.2728 - val_accuracy: 0.9205
    Epoch 16/30
    115/115 [==============================] - 6s 50ms/step - loss: 0.0380 - accuracy: 0.9981 - val_loss: 0.2779 - val_accuracy: 0.9148
    Epoch 17/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0352 - accuracy: 0.9976 - val_loss: 0.2880 - val_accuracy: 0.9119
    Epoch 18/30
    115/115 [==============================] - 6s 53ms/step - loss: 0.0310 - accuracy: 0.9978 - val_loss: 0.2842 - val_accuracy: 0.9176
    Epoch 19/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0296 - accuracy: 0.9992 - val_loss: 0.2888 - val_accuracy: 0.9176
    Epoch 20/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0258 - accuracy: 0.9984 - val_loss: 0.2847 - val_accuracy: 0.9134
    Epoch 21/30
    115/115 [==============================] - 6s 49ms/step - loss: 0.0238 - accuracy: 0.9997 - val_loss: 0.2812 - val_accuracy: 0.9247
    Epoch 22/30
    115/115 [==============================] - 6s 50ms/step - loss: 0.0233 - accuracy: 0.9995 - val_loss: 0.2874 - val_accuracy: 0.9176
    Epoch 23/30
    115/115 [==============================] - 6s 50ms/step - loss: 0.0204 - accuracy: 0.9989 - val_loss: 0.2882 - val_accuracy: 0.9148
    Epoch 24/30
    115/115 [==============================] - 6s 52ms/step - loss: 0.0195 - accuracy: 0.9986 - val_loss: 0.2863 - val_accuracy: 0.9190
    Epoch 25/30
    115/115 [==============================] - 6s 50ms/step - loss: 0.0173 - accuracy: 1.0000 - val_loss: 0.2783 - val_accuracy: 0.9219
    Epoch 26/30
    115/115 [==============================] - 6s 50ms/step - loss: 0.0156 - accuracy: 0.9997 - val_loss: 0.2891 - val_accuracy: 0.9190
    Epoch 27/30
    115/115 [==============================] - 6s 51ms/step - loss: 0.0153 - accuracy: 0.9997 - val_loss: 0.2846 - val_accuracy: 0.9190
    Epoch 28/30
    115/115 [==============================] - 6s 51ms/step - loss: 0.0159 - accuracy: 1.0000 - val_loss: 0.2835 - val_accuracy: 0.9148
    Epoch 29/30
    115/115 [==============================] - 6s 50ms/step - loss: 0.0137 - accuracy: 0.9997 - val_loss: 0.2856 - val_accuracy: 0.9205
    Epoch 30/30
    115/115 [==============================] - 6s 51ms/step - loss: 0.0123 - accuracy: 0.9997 - val_loss: 0.2947 - val_accuracy: 0.9162


## Plot learning curves

When we plot the accuracy and loss curves for both our training and test sets we can see the model converges very quickly. This is one of the very reasons we use transfer learning.


```python
acc  = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.6, 1.01])
plt.plot([EPOCHS-1,EPOCHS-1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([-0.01, 1.5])
plt.plot([EPOCHS-1,EPOCHS-1])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

```


![png](/assets/images/basic-transfer-learning/output_20_0.png)


## Evaluation of predictions

Here we evaluate our predictions. In particular we will observe how well our model adapts to the new classes not included in the training of the feature vector and how well it predicts dogs vs cats. We measure the accuracy on each subset of the images. W also plot a sample of the predicted images where a false prediction has a red label with the true label in parentheses. 

We can see that accuracy is much higher on breeds already familiar to the feature vector. No surprises here. What might be surprising is how well the model performed on the 14 breeds it had not seen 
before. The final evaluation was 88.5% accuracy on such breeds. 


```python
def plotPredictedImages(batch, pred, title):
    img, label = batch
    
    plt.figure(figsize=(10,10))
    plt.suptitle(title, fontsize='xx-large', y=1.05)
    plt.subplots_adjust(hspace=0.5)
    
    for n in range(12):
        
        plt.subplot(3,4,n+1)
        plt.imshow(img[n])
        p = pred[n]
        l = label[n].numpy()

        if pred[n] == label[n]:
            plt.title(feat_dict[pred[n]], color="green")
        else:
            plt.title(feat_dict[p] + "\n" + " (" + feat_dict[l] + ")", color="red")
        
        plt.axis('off')    

    plt.tight_layout()
    plt.show()

all_accuracy = model.evaluate(test, verbose = 0)[1]
title = 'All classes\nAccuracy: {0:.2f}%'.format(all_accuracy*100)

sample_test_batch = next(iter(test))
predictions = model.predict_classes(sample_test_batch)
plotPredictedImages(sample_test_batch, predictions, title) 

NEW_CLASSES_STR = [
    "Sphynx",
    "British Shorthair",
    "Maine Coon",
    "Abyssinian",
    "Bengal",
    "Egyptian Mau",
    "Russian Blue",
    "Ragdoll",
    "Birman",
    "Bombay",
    "havanese",
    "leonberger",
    "american bulldog",
    "shiba inu"
]

inv_feat_dict = {v: k for k, v in feat_dict.items()}

NEW_CLASSES = tf.constant(
    [inv_feat_dict[i] for i in NEW_CLASSES_STR], 
    dtype=tf.int64)

test_new = test.unbatch().filter(
    lambda image, label: tf.reduce_any(tf.equal(label, NEW_CLASSES))
).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

test_old = test.unbatch().filter(
    lambda image, label: tf.reduce_all(tf.not_equal(label, NEW_CLASSES))
).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

CATS_STR = list(filter(lambda x: x[0].isupper() , list(inv_feat_dict.keys())))

CATS = tf.constant(
    [inv_feat_dict[i] for i in CATS_STR], 
    dtype=tf.int64)

test_cats = test.unbatch().filter(
    lambda image, label: tf.reduce_any(tf.equal(label, CATS))
).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dogs = test.unbatch().filter(
    lambda image, label: tf.reduce_all(tf.not_equal(label, CATS))
).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

new_accuracy = model.evaluate(test_new, verbose = 0)[1]
title = 'New classes only\nAccuracy: {0:.2f}%'.format(new_accuracy*100)

sample_test_new_batch = next(iter(test_new))
predictions_new = model.predict_classes(sample_test_new_batch)
plotPredictedImages(sample_test_new_batch, predictions_new, title) 

old_accuracy = model.evaluate(test_old, verbose = 0)[1]
title = 'Old classes only\nAccuracy: {0:.2f}%'.format(old_accuracy*100)

sample_test_old_batch = next(iter(test_old))
predictions_old = model.predict_classes(sample_test_old_batch)
plotPredictedImages(sample_test_old_batch, predictions_old, title) 

cats_accuracy = model.evaluate(test_cats, verbose = 0)[1]
title = 'Cats only\nAccuracy: {0:.2f}%'.format(cats_accuracy*100)

sample_test_cats_batch = next(iter(test_cats))
predictions_cats = model.predict_classes(sample_test_cats_batch)
plotPredictedImages(sample_test_cats_batch, predictions_cats, title) 

dogs_accuracy = model.evaluate(test_dogs, verbose = 0)[1]
title = 'Dogs only\nAccuracy: {0:.2f}%'.format(dogs_accuracy*100)

sample_test_dogs_batch = next(iter(test_dogs))
predictions_dogs = model.predict_classes(sample_test_dogs_batch)
plotPredictedImages(sample_test_dogs_batch, predictions_dogs, title)
```

![png](/assets/images/basic-transfer-learning/output_23_0.png)


![png](/assets/images/basic-transfer-learning/output_26_0.png)


![png](/assets/images/basic-transfer-learning/output_27_0.png)


![png](/assets/images/basic-transfer-learning/output_28_0.png)


![png](/assets/images/basic-transfer-learning/output_29_0.png)


## Conclusion

In a short period of time with a comparatively simple architecture we were able to develop a model on a new dataset that converged quickly and got results of 88.5% on classes it had not seen before and 91.11% overall. I hope this demonstrates the value of transfer learning. 

