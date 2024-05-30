---
title: "Thai-to-English Translation"
date: 2024-04-19
categories:
  - nlp
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
---
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1KAnB2GcQIr3-pbtz-qLUFhPKu7S1jPch?usp=sharing)

Machine translation is a pivotal technology for bridging language gaps and enabling effective global communication. Translating from English to Thai is particularly challenging due to substantial differences in grammar, syntax, and vocabulary. Thai, being a tonal language with intricate word formations and no spaces between words, requires advanced handling in the translation process. In this project, we aim to develop a high-quality translation model using a Transformer-based architecture. Transformers represent the cutting edge in natural language processing (NLP), utilizing self-attention mechanisms to capture contextual relationships within sentences more effectively than previous models. By leveraging the Keras NLP library, we can efficiently process text, tokenize language data, and build sophisticated models to tackle the complexities of English-to-Thai translation.

# Setup

```python
!pip install -q --upgrade datasets keras keras-nlp pyter3 pythainlp sacrebleu tensorflow tensorflow-text
```

```python
# Standard library imports
import random
import re
import unicodedata

# Tensorflow and related libraries
import tensorflow as tf
import tensorflow.data as tf_data
import keras
import keras_nlp
from keras import ops
from tensorflow_text.tools.wordpiece_vocab import (
    bert_vocab_from_dataset as bert_vocab,
)

# Other third-party libraries
import sacrebleu
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize as tha_segment
from pythainlp.util import normalize as tha_normalize
```


```python
BATCH_SIZE = 64
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 32000
THA_VOCAB_SIZE = 32000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8
```

# Data Preparation

## Load

The dataset used in this project is part of the OPUS project, a growing collection of open parallel corpora for machine translation. It contains around 1 million English-Thai sentence pairs, providing substantial training data. Sourced from various domains like websites, subtitles, and religious texts, the dataset ensures a diverse range of topics and language usage. The data is easily accessible via the Hugging Face data loader, making it convenient for integration into machine translation workflows. Additionally, the data comes pre-split into training, validation, and test sets, saving us a step.

```python
dataset = load_dataset("opus100", "en-th")
```

## Normalize

Normalization and cleaning are essential for preparing both English and Thai text for tokenization and model training. These steps ensure the text is in a consistent format suitable for further processing.

For English, this includes Unicode normalization to standardize characters, converting text to lowercase, expanding common contractions, removing special characters and punctuation, and normalizing whitespace.

For Thai, the process involves Unicode normalization, using the `pythainlp` library to standardize Thai text (as Thai often has variations in character representation), removing punctuation marks (while preserving essential special characters to avoid damaging the text), and using the `pythainlp` library again to segment Thai words (since there is typically no whitespace between words).


```python
def normalize_thai_text(text):
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)

    # Normalize Thai text using pythainlp
    text = tha_normalize(text)

    # Remove special characters and punctuation (can't remove all special characters for Thai!)
    punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    text = re.sub(punctuation_pattern, '', text)

    # Segment words and normalize whitespace
    text = re.sub(r'\s+', '', text).strip()
    text = ' '.join(tha_segment(text))

    return text

def normalize_english_text(text):
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)

    # Convert to lowercase
    text = text.lower()

    # Expand contractions (example for common contractions)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_dataset(dataset):
  data = []
  for example in dataset:
      en = example['translation']["en"]
      th = example['translation']["th"]
      en = normalize_english_text(en)
      th = normalize_thai_text(th)
      data.append((en, th))
  return data

train_pairs = clean_dataset(dataset["train"])
val_pairs = clean_dataset(dataset["validation"])
test_pairs = clean_dataset(dataset["test"])

print("Samples: ")
for _ in range(5):
    print(random.choice(train_pairs))
```

    Samples: 
    ('ben lee was one of her students', 'เบ็น ลี เป็นหนึ่ง ใน นักเรียน ของ เธอ')
    ('i am really sorry', 'ฉัน ขอโทษ จริงๆ ๆ คะ')
    ('oh i am sure they will', 'โอ้ ฉัน เชื่อ ว่า พวกเขา จะ มา')
    ('took you out last week', 'ก็ พา ไป อาทิตย์ ที่แล้ว ไง')
    ('she will be here in two minutes find her toorop', 'หา ประตู ที่ เปิด ทิ้ง ไว้ คุณ เป็น คนเดียว ที่จะ คุ้มครอง หล่อน ได้ ใน ตอนนี้')


## Tokenize

Tokenization breaks down text into smaller units and encodes them into numerical representations, which is essential for natural language processing tasks. We use WordPiece tokenization to split text into subword units, capturing common prefixes, suffixes, and roots. This method involves building a WordPiece vocabulary from the dataset, breaking sentences into words and then into subwords.

The generated vocabularies include reserved tokens such as `[PAD]`, `[UNK]`, `[START]`, and `[END]`, which serve special purposes: `[PAD]` is used for padding sequences to a uniform length, `[UNK]` represents unknown words, `[START]` marks the beginning of a sequence, and `[END]` marks the end. The tokenizers created with these vocabularies can handle sequences up to a maximum length, ensuring each sentence is represented as a sequence of subword tokens in numerical form for the neural network. This approach helps manage out-of-vocabulary words, standardize input lengths, and reduce vocabulary size, making the model more efficient.


```python
def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf_data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)

tha_samples = [text_pair[1] for text_pair in train_pairs]
tha_vocab = train_word_piece(tha_samples, THA_VOCAB_SIZE, reserved_tokens)

print("English Tokens: ", eng_vocab[1000:1010])
print("Thai Tokens: ", tha_vocab[1000:1010])
```

    English Tokens:  ['birthday', 'hide', 'ride', 'history', 'rather', 'spend', 'ground', 'hotel', 'situation', 'state']
    Thai Tokens:  ['เป็นไปได้', 'ข้างใน', 'ดึง', 'ทั้งคู่', 'ครึ่ง', 'ว่ะ', 'ข้อ', 'บี', 'ห้า', 'สถานที่']



```python
eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=eng_vocab, lowercase=False
)
tha_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=tha_vocab, lowercase=False
)

eng_input_ex = train_pairs[0][0]
eng_tokens_ex = eng_tokenizer.tokenize(eng_input_ex)
print("English sentence: ", eng_input_ex)
print("Tokens: ", eng_tokens_ex)
print(
    "Recovered text after detokenizing: ",
    eng_tokenizer.detokenize(eng_tokens_ex).numpy().decode("utf-8"),
)

print()

tha_input_ex = train_pairs[0][1]
tha_tokens_ex = tha_tokenizer.tokenize(tha_input_ex)
print("Thai sentence: ", tha_input_ex)
print("Tokens: ", tha_tokens_ex)
print(
    "Recovered text after detokenizing: ",
    tha_tokenizer.detokenize(tha_tokens_ex).numpy().decode("utf-8"),
)
```

    English sentence:  pray to god sahib
    Tokens:  tf.Tensor([ 1777   195   309 10640  4099  1352], shape=(6,), dtype=int32)
    Recovered text after detokenizing:  pray to god sahib
    
    Thai sentence:  สวดมนต์ สิ เจ้านาย
    Tokens:  tf.Tensor([4494  310 1497], shape=(3,), dtype=int32)
    Recovered text after detokenizing:  สวดมนต์ สิ เจ้านาย


## Format Dataset for Training

After tokenization, we prepare the dataset for training by pairing and formatting the tokenized English and Thai sentences. At each training step, the model predicts the next target word using the source sentence and the previously predicted target words.

The Transformer model requires two inputs: the tokenized source sentence (English) and the partially produced target sentence (Thai) up to the current word, which are both used to predict the next word. The target sentence itself is what the model aims to predict.

We add special reserved tokens to both the tokenized source and target sentences to pad the sequences and mark the start and end.

These formatted pairs are then batched and mapped into TensorFlow datasets, ensuring the data is correctly structured and ready for efficient neural network training.


```python
def format_batch(eng, tha):
    batch_size = ops.shape(tha)[0]

    eng = eng_tokenizer(eng)
    tha = tha_tokenizer(tha)

    # Pad `eng` to `MAX_SEQUENCE_LENGTH`.
    eng_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=eng_tokenizer.token_to_id("[PAD]"),
    )
    eng = eng_start_end_packer(eng)

    # Add special tokens (`"[START]"` and `"[END]"`) to `tha` and pad it as well.
    tha_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=tha_tokenizer.token_to_id("[START]"),
        end_value=tha_tokenizer.token_to_id("[END]"),
        pad_value=tha_tokenizer.token_to_id("[PAD]"),
    )
    tha = tha_start_end_packer(tha)

    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": tha[:, :-1],
        },
        tha[:, 1:],
    )


def format_dataset(pairs):
    eng_texts, tha_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    tha_texts = list(tha_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, tha_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_batch, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = format_dataset(train_pairs)
val_ds = format_dataset(val_pairs)
```

# Training

## Building Model Architecture

To create a Transformer-based architecture for machine translation, we start with an embedding layer to create a vector for every token in our input sequence, which can be initialized randomly. We also need a positional embedding layer to encode word order in the sequence. The `TokenAndPositionEmbedding` layer in KerasNLP handles both tasks for us.

The source sequence is passed to the `TransformerEncoder`, producing a new representation. This representation, along with the target sequence so far, is passed to the `TransformerDecoder`, which predicts the next words in the target sequence. The final layer applies a softmax activation to predict the most likely Thai vocabulary tokens, and dropout is applied as a form of regularization.

Causal masking ensures the `TransformerDecoder` only uses information from target tokens seen so far when predicting the next token, preventing it from seeing future tokens.

We also need to mask padding tokens (`[PAD]`). Setting the `mask_zero` argument of the `TokenAndPositionEmbedding` layer to True ensures this masking is propagated to all subsequent layers.


```python
# Encoder
encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=ENG_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=THA_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
)(decoder_inputs)

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(THA_VOCAB_SIZE, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)

transformer.summary()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "transformer"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)              </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">        Param # </span>┃<span style="font-weight: bold"> Connected to           </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ encoder_inputs            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)           │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)              │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ token_and_position_embed… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │      <span style="color: #00af00; text-decoration-color: #00af00">8,202,240</span> │ encoder_inputs[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TokenAndPositionEmbeddi…</span> │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ decoder_inputs            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)           │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)              │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ transformer_encoder       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │      <span style="color: #00af00; text-decoration-color: #00af00">1,315,072</span> │ token_and_position_em… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TransformerEncoder</span>)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ functional_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32000</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">18,004,992</span> │ decoder_inputs[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],  │
│                           │                        │                │ transformer_encoder[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
└───────────────────────────┴────────────────────────┴────────────────┴────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">27,522,304</span> (104.99 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">27,522,304</span> (104.99 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>

## Training Model

We compile the Transformer model with an appropriate loss function and optimizer, then fit it to the prepared training dataset.

The training process runs for 10 epochs, during which the model learns to translate English sentences into Thai by minimizing translation errors. We monitor the training using accuracy on the validation dataset to prevent overfitting and ensure the model's generalizability. We use accuracy over typical NLP metrics because it is computationally efficient.


```python
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
```

    Epoch 1/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m472s[0m 28ms/step - accuracy: 0.8250 - loss: 1.2532 - val_accuracy: 0.8283 - val_loss: 1.0962
    Epoch 2/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m436s[0m 28ms/step - accuracy: 0.8441 - loss: 1.0172 - val_accuracy: 0.8342 - val_loss: 1.0608
    Epoch 3/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m471s[0m 30ms/step - accuracy: 0.8486 - loss: 0.9748 - val_accuracy: 0.8361 - val_loss: 1.0363
    Epoch 4/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m486s[0m 31ms/step - accuracy: 0.8516 - loss: 0.9429 - val_accuracy: 0.8379 - val_loss: 1.0203
    Epoch 5/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m463s[0m 30ms/step - accuracy: 0.8541 - loss: 0.9204 - val_accuracy: 0.8388 - val_loss: 1.0093
    Epoch 6/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m456s[0m 29ms/step - accuracy: 0.8561 - loss: 0.9032 - val_accuracy: 0.8407 - val_loss: 0.9967
    Epoch 7/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m447s[0m 29ms/step - accuracy: 0.8581 - loss: 0.8875 - val_accuracy: 0.8422 - val_loss: 0.9884
    Epoch 8/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m458s[0m 29ms/step - accuracy: 0.8599 - loss: 0.8743 - val_accuracy: 0.8426 - val_loss: 0.9825
    Epoch 9/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m447s[0m 29ms/step - accuracy: 0.8614 - loss: 0.8636 - val_accuracy: 0.8431 - val_loss: 0.9844
    Epoch 10/10
    [1m15625/15625[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m443s[0m 28ms/step - accuracy: 0.8627 - loss: 0.8543 - val_accuracy: 0.8425 - val_loss: 0.9884


# Evaluation

## Translation Method

We create a method to use the trained Transformer model to translate English sentences into Thai. Given an English input sentence, we tokenize it and feed it into the encoder, along with the target token `[START]`. The decoder then outputs probabilities for the next token. Using `keras_nlp.samplersGreedySampler`, we select the most likely next token at each step, based on the tokens generated so far. This process continues until the `[END]` token is produced. The generated tokens are then detokenized to form the final translated Thai sentence. This method ensures that the model produces fluent and coherent translations based on the patterns learned from the training data.


```python
def decode_sequences(input_sentences):
    batch_size = 1

    eng_start_end_packer = keras_nlp.layers.StartEndPacker(
      sequence_length=MAX_SEQUENCE_LENGTH,
      pad_value=eng_tokenizer.token_to_id("[PAD]"),
    )
    eng_tokens = eng_tokenizer(input_sentences)
    eng_inputs = eng_start_end_packer(eng_tokens)
    eng_inputs = ops.cast(eng_inputs, tf.float32)


    # Define a function that outputs the next token's probability given the
    # input sequence.
    def next(prompt, cache, index):
        logits = transformer([eng_inputs, prompt])[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        return logits, hidden_states, cache

    # Build a prompt of length 40 with a start token and padding tokens.
    length = 40
    start = ops.full((batch_size, 1), tha_tokenizer.token_to_id("[START]"))
    pad = ops.full((batch_size, length - 1), tha_tokenizer.token_to_id("[PAD]"))
    prompt = ops.concatenate((start, pad), axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        stop_token_ids=[tha_tokenizer.token_to_id("[END]")],
        index=1,  # Start sampling after start token.
    )

    generated_sentences = tha_tokenizer.detokenize(generated_tokens)
    return generated_sentences
```

## Qualitative Evaluation

To assess translation quality, we compare model-generated Thai sentences with reference translations from the dataset. This involves manually inspecting translations for fluency, accuracy, and naturalness, which works best if you can read both languages. Alternatively, you can compare the translations to those from existing translators like Google Translate or ChatGPT. This evaluation helps identify the model's strengths and weaknesses, providing insights into areas for improvement. Such qualitative assessment is crucial for understanding the model's practical effectiveness beyond numerical metrics.


```python
test_eng_texts = [pair[0] for pair in test_pairs]
for i in range(10):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequences([input_sentence])
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    print(f"** Example {i+1} **")
    print(input_sentence)
    print(translated)
    print()
```

    ** Example 1 **
    yeah
    - ใช่
    
    ** Example 2 **
    you jealous because we are fitting in with your cool friends
    คุณ อิจฉา เพราะ เรา จะ อยู่ กับ เพื่อน ของ คุณ
    
    ** Example 3 **
    it is your job to see they do not mine is to protect the integrity of this op
    มัน เป็น งาน ของ คุณ ที่ ไม่ ได้ ทำ เพื่อ ปกป้อง สิทธิ ของ คุณ
    
    ** Example 4 **
    ye eun
    - อึน อึน
    
    ** Example 5 **
    to you i will be just a fossil from a long time ago
    เพื่อ เป็น แค่ ซาก ของ ลูก
    
    ** Example 6 **
    but it was not something you want to discuss
    แต่ มัน ไม่ ใช่ เรื่อง ที่ คุณ ต้องการ คุย
    
    ** Example 7 **
    lamentable
    ไม่ มี ความ ไม่ ดี
    
    ** Example 8 **
    miguel really did it and i taught him how
    มิ เกล ทำ มัน และ ฉัน สอน เขา ว่า ไง
    
    ** Example 9 **
    what are you going to do about my lips
    คุณ จะ ทำ ยังไง กับ ปาก ฉัน ?
    
    ** Example 10 **
    no there is no other ayanami but you
    ไม่ มี ไม่ มี อา ยา น่า แต่ เธอ
    


## Quantitative Evaluation

We calculate the ChrF (Character F-score) metric, which measures the similarity between the model-generated translations and the reference translations. This metric considers character n-gram precision and recall, providing a balanced assessment of translation accuracy and fluency. By comparing the generated translations to the reference translations on a large test set, the ChrF score quantifies the model's overall translation quality, enabling objective evaluation and comparison with other models or baselines.

We use the ChrF metric instead of ROUGE or BLEU because ChrF evaluates translation quality based on character n-grams rather than word n-grams. This approach provides several advantages:

- Granularity: ChrF operates at the character level, making it more sensitive to small differences and more suitable for languages with complex morphology, like Thai.
- Language Agnostic: ChrF is better at handling languages with different scripts and tokenization challenges, such as Thai, where word boundaries are not always clear.
- Balanced Assessment: ChrF considers both precision and recall of character n-grams, offering a balanced measure of translation accuracy and fluency.

These characteristics make ChrF a more appropriate choice for evaluating translations involving languages with unique linguistic features, providing a more nuanced and accurate assessment than ROUGE or BLEU.


```python
def calculate_overall_chrf(references, candidates):
    chrf_scores = [sacrebleu.sentence_chrf(candidate, [ref]).score for ref, candidate in zip(references, candidates)]
    overall_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0
    return overall_chrf


references = []
candidates = []

for test_pair in test_pairs[:100]:
    input_sentence = test_pair[0]
    reference_sentence = test_pair[1]

    translated_sentence = decode_sequences([input_sentence])
    translated_sentence = translated_sentence.numpy()[0].decode("utf-8")
    translated_sentence = (
        translated_sentence.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )

    reference_sentence = ''.join(reference_sentence.split())
    translated_sentence = ''.join(translated_sentence.split())

    references.append(reference_sentence)
    candidates.append(translated_sentence)

# Calculate overall ChrF score
overall_chrf = calculate_overall_chrf(references, candidates)
print(f'Overall ChrF score: {overall_chrf}')
```

    Overall ChrF score: 28.038110243006408
