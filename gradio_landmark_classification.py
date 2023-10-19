import os
import numpy as np
import gradio as gr
import pandas as pd
import tensorflow as tf
import PIL.Image as Image
import tensorflow_hub as hub
import matplotlib.pyplot as plt

TF_MODEL_URL = "https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_africa_V1/1"
LABEL_MAP_URL = "https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_africa_V1_label_map.csv"
IMAGE_SHAPE = (321, 321)

classifier = tf.keras.Sequential([hub.KerasLayer(
                TF_MODEL_URL,
                input_shape = (*IMAGE_SHAPE, 3 ),
                output_key = "predictions:logits"
)])


df = pd.read_csv(LABEL_MAP_URL)

df.head()

label_map = dict(zip(df.id, df.name))
[(k,v) for k,v in label_map.items()][:5]

BASE_DIR = "C:/Users/chine/Desktop/PythonScripts/captum_tutorial/landmark-images/Images"
test_images = os.listdir(BASE_DIR)

img =  Image.open(BASE_DIR + "/" + test_images[2]).resize(IMAGE_SHAPE)
img

img = np.array(img)/255.0
img.shape
img = img[np.newaxis, ...]
img.shape

result = classifier.predict(img)
result.shape

label_map[np.argmax(result)]

def classify_img(image):
    img = np.array(image)/255.0
    img = img[np.newaxis, ...]
    prediction = classifier.predict(img)
    return label_map[np.argmax(prediction)]
image = gr.inputs.Image(shape=(321, 321))
label = gr.outputs.Label(num_top_classes=1)

app = gr.Interface(
    classify_img,
    image,
    label,
    capture_session=True
)

app.launch(share=True)

