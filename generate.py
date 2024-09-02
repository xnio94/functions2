import glob
import subprocess
import urllib

from flask import send_file, jsonify

from functions.dynamic_import import dynamic_import




import time
import numpy as np
import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# print("Model loading Model....................")
# start_time = time.time()
# loaded_model = tf.saved_model.load('efficientnetb0_saved_model')
# end_time = time.time()
# print(f"Model loading time: {end_time - start_time} seconds")
#
# # To use the loaded model, you may need to wrap it in a Keras Model again
# infer = loaded_model.signatures["serving_default"]





from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet')
target_size=(224, 224)
# Remove the top layer (classification layer) to get the feature vectors
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


def get_image_embedding(img_path):
    # Load the image with the target size of 224x224
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    print(img)
    # print(len(img))
    img_array = image.img_to_array(img)
    print(img_array.shape)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)
    img_array = preprocess_input(img_array)

    return model.predict(img_array)


    print(img_array.shape)

    # Convert the image array to a tensor
    img_tensor = tf.convert_to_tensor(img_array)
    print(img_tensor.shape)

    # Use the inference function to get the embedding
    embedding = infer(img_tensor)['top_dropout']
    print(embedding.shape)
    return embedding

def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

def test_f():
    print("embedding1....................")
    embedding1 = get_image_embedding('1.jpg')
    print("embedding2....................")
    embedding2 = get_image_embedding('2.jpg')
    v = -1
    cosine_sim = compute_cosine_similarity(embedding1[:,:v], embedding2[:,:v])
    return cosine_sim

import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def check_if_scanned(clip_id):
    doc_ref = db.collection('clips').document('clips')
    doc = doc_ref.get()
    # doc_ref.set({
    #     'scanned': False
    # })
    # if doc.exists:
    scanned = doc.to_dict().get(clip_id)
    if scanned is None:
        return False
    return scanned

def check_if_scanned_bulk(clip_ids):
    doc_ref = db.collection('clips').document('clips')
    doc = doc_ref.get()
    scanned_dict = doc.to_dict()
    return [scanned_dict.get(clip_id, False) for clip_id in clip_ids]

def generate(request):
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    }
    # log_state = dynamic_import('log_state')


    request_json = request.get_json()
    if request_json and 'test_f' in request_json:
        x = test_f()
        return jsonify({'test_f': float(x)}), 200, headers


    if request_json and 'id' in request_json:
        id = request_json['id']
        scanned = check_if_scanned(id)
        return jsonify({'scanned': scanned}), 200, headers

    if request_json and 'ids' in request_json:
        ids = request_json['ids']
        scanned_list = check_if_scanned_bulk(ids)
        return jsonify({'scanned_list': scanned_list}), 200, headers

    #
    #     data = type('', (), request_json)()
    #     for attribute in request_json.keys():
    #         data.__setattr__(attribute, request_json[attribute])
    #
    #     if requested_field == "all":
    #         def set_field(field_name, data):
    #             setattr(data, field_name, generate_field(field_name, data))
    #
    #         # data = json.dumps(data.__dict__)
    #         # return data, 200, headers
    #
    #         return (
    #             jsonify({
    #                         'error': requested_field + ' not found, check documentation for valid generation'}),
    #             200,
    #             headers)
    #
    #     try:
    #         result = generate_field(requested_field, data)
    #         return jsonify({requested_field: result}), 200, headers
    #     except Exception as e:
    #         return (jsonify({'error': str(
    #             e) + ' ===> check that all required attributes are sent for this generation'}),
    #                 200,
    #                 headers)
    # #
    # else:
    #     return (jsonify({'error': 'generate field not found'}),
    #             200,
    #             headers)
