import pprint
from random import random

import cv2
import numpy as np
import requests
from mtcnn import MTCNN
from pymongo import MongoClient
from tensorflow.keras.preprocessing import image as image_preprocess

import VGGFace

client = MongoClient(host='localhost', port=27017)
db = client.facima 
db.authenticate('facima', '09371880706')
collection = db.facima
collection.remove({})

probe_directory = "http://localhost/images/probe/"
probe_images = requests.get(probe_directory + "images.txt").text.split()
print("The number of probe images is equal to " + str(len(probe_images)))


gallery_directory = "http://localhost/images/gallery/"
gallery_images = requests.get(gallery_directory + "images.txt").text.split()
print("The number of gallery images is equal to " + str(len(gallery_images)))

FACE_DETECTOR = MTCNN()
FACE_REPRESENTATION = VGGFace.loadModel()


def face_detection(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = FACE_DETECTOR.detect_faces(img_rgb)
    
def preprocess(img):
    target_size=(224, 224)
    img = cv2.resize(img, target_size)
    img_pixels = image_preprocess.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 

    return img_pixels


def cosine(emb1, emb2):
    emb12 = np.sum(np.multiply(emb1 , emb2))
    emb11 = np.sum(np.sqrt(emb1 ** 2))
    emb22 = np.sum(np.sqrt(emb2 ** 2))
    distance = emb12 / (emb11 * emb22)

    return float(distance)

def match(probe, gallery):
    prob_image = face_detection(probe)
    prob_image = preprocess(probe)
    prob_image_representation = FACE_REPRESENTATION.predict(prob_image)[0,:]

    gallery_image = face_detection(gallery)
    gallery_image = preprocess(gallery_image)
    gallery_image_representation = FACE_REPRESENTATION.predict(gallery_image)[0,:]

    distance = cosine(prob_image_representation, gallery_image_representation)

    return distance

for probe_image in probe_images:
    temp = {}
    similarities = []
 
    p_image = requests.get(probe_directory + probe_image).content
    p_image_decoded = cv2.imdecode(np.frombuffer(p_image, np.uint8), -1)
    for gallery_image in gallery_images:

        g_image = requests.get(gallery_directory + gallery_image).content
        g_image_decoded = cv2.imdecode(np.frombuffer(g_image, np.uint8), -1)
    	
        temp[gallery_image.split(".")[0]] = match(p_image_decoded, g_image_decoded)
   
    temp = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1], reverse=True)}
 
    for key in temp:
        similarities.append({key: temp[key]})
  
    candidate_list = {probe_image.split(".")[0]: similarities}
    inserted_id = collection.insert_one(candidate_list).inserted_id


for result_list in collection.find():
    pprint.pprint(result_list)

