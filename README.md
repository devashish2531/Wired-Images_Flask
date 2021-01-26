# Wired-Images_Flask
# [Python Notebook Repository](https://github.com/devashish2531/Wired_Images-AI_Image_Captioning_Bot/)

Flask Web Application for [Wired_Images-AI_Image_Captioning_Bot](https://github.com/devashish2531/Wired_Images-AI_Image_Captioning_Bot/).
With the help of this project, machine can generate its own captions related to the image and present it in an elegant manner. 
The model uses Flickr8k dataset for training data and uses LSTM for training and CNN for extracting the image features.

## Dataset
Flickr30k Image dataset has been used for training the model.
This dataset contains 8000 images and there are 5 captions for each image.

## The Model made contains:

1) Image model: for reducing image of high dimensions into selected features.  
2) Language model: for creating output as embeddings to reduce complexity of model. 
3) Final model: which will concatenate results of both this model and will use LSTM layer and Time Distributed layer for doing final_predicttions.

## Screenshots
![FaceableCapture2](https://i.imgur.com/Z6bzOlg.jpg)
![FaceableCapture3](https://i.imgur.com/eEaXMpQ.jpg)
![FaceableCapture3](https://i.imgur.com/lybydHX.jpg)


