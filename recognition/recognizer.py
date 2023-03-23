from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from gtts import gTTS
from IPython.display import Audio


import os
from pickle import load
import numpy as np
from PIL import Image

def extract_features(filename, model):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)

        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
  return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def test_model(path):
  # img_path = '/content/drive/MyDrive/imag_caption/Flickr8k_Dataset/Flicker8k_Dataset' + '/' + path
  img_path = path
  max_length = 22

  token_map = os.path.dirname(__file__) +'\\token_map.p'
  tokenizer = load(open(token_map,"rb"))

  model_8 = os.path.dirname(__file__) +'\\Xception3K_model.h5'
  model = load_model(model_8)
  xception_model = Xception(include_top=False, pooling="avg")

  photo = extract_features(img_path, xception_model)

  description = generate_desc(model, tokenizer, photo, max_length)

  return description[5:-4]

  sound_file =audio(description)
  return sound_file



