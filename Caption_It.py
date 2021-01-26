import pandas as pd
import numpy as np
import keras
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

model = load_model('./model_weights/model_9.h5')
model.make_predict_function()
model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))
model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
model_resnet.make_predict_function()


with open('./storage/word_to_idx.pkl','rb') as w2i:
    word_to_idx = pickle.load(w2i)
with open('./storage/idx_to_word.pkl','rb') as i2w:
    idx_to_word = pickle.load(i2w)

def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)   
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector

def predict_caption(photo):
    max_len=35
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


def caption_this_image(image):
    enc=encode_image(image)
    caption = predict_caption(enc)
    return caption






