from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import warnings
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
from fastai import *
from fastai.vision import *

import numpy as np
import os, time, sys, gc
from mrcnn.model import MaskRCNN
from mrcnn import visualize
import skimage.io
from src.config import *
from src.dataset import *

class_names = ['BG','cow']

cfg = CattlePredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='models/', config=cfg)

print("[INFO] Loading Weights...")
model.load_weights('models/mask_rcnn_cattle_config_0004.h5', by_name=True)
image_name = "0.jpg"
image = skimage.io.imread(image_name)
results = model.detect([image], verbose=1)

cred = credentials.Certificate("sih2020-e29b2-firebase-adminsdk-abixl-1367d4ad1b.json")
firebase_admin.initialize_app(cred,{
'databaseURL' : 'https://sih2020-e29b2.firebaseio.com'
})
learn = load_learner('data')

warnings.filterwarnings("ignore")
app = Flask(__name__)

def getCowCount(filename):
    global model
    image = skimage.io.imread(filename)
    start_time = time.time()
    results = model.detect([image], verbose=0)
    print("\n[INFO]Time taken : \n", time.time() - start_time)
    r = results[0]
    results = list()
    for i,score in enumerate(r['scores']):
        if score > 0.95:
            results.append(score)
    numOfCows = len(results)
    return numOfCows

def getCowClassificationOutput(filename):
    global learn
    img = open_image(filename)
    pred_class,pred_idx,outputs = learn.predict(img)
    if pred_class.obj=="stray_cattle":
        return True
    else:
        return False

@app.route("/complaint", methods=['POST','GET'])
def sendResult():
    try:
        d = request.get_json()
        print("awsm fam")
        picnp = np.fromstring(base64.b64decode(d["img"]), dtype=np.uint8)
        img = cv2.imdecode(picnp, 1)
        cv2.imwrite("trial.jpg",img)
        print("\n\n")
        cow_classification_output = getCowClassificationOutput("trial.jpg")
        if cow_classification_output:
            number = getCowCount("trial.jpg")
            root = db.reference('Complaints/')
            auths = db.reference('Authority/')
            authorities = auths.get()
            incharge_id = "no incharge"
            for k,v in authorities.items():
                if v["assigned_area"] == d['locality']:
                    incharge_id = v["id"]
                    break
            d["incharge_id"] = incharge_id
            d['numOfCows'] = number
            root.child(d['cid']).set(d)
            return "true"
        else:
            return "false"

    except Exception as e:
        print("ERROR! ",e)
        return "false"

@app.route("/resolve", methods=['POST','GET'])
def sendResult2():
    try:
        d = request.get_json()
        picnp = np.fromstring(base64.b64decode(d["img"]), dtype=np.uint8)
        img = cv2.imdecode(picnp, 1)
        cv2.imwrite("trial.jpg",img)
        # cv2.imwrite("image2.png",img)
        cow_classification_output = getCowClassificationOutput("trial.jpg")
        if not cow_classification_output:
            print("getting db ref..")
            root = db.reference('Complaints/')
            comps = root.get()
            for k,v in comps.items():
                if d["cid"] == v["cid"]:
                    db.reference("Resolved/").child(d['cid']).set(d)
                    root.child(d['cid']).delete()
                    break
            return "true"
        else:
            return "false"
    except Exception as e:
        print("ERROR! ",e)
        return "false"

@app.route("/authresolve", methods=['POST','GET'])
def sendResult3():
    try:
        d = request.get_json()
        print("getting db ref..")
        root = db.reference('Complaints/')
        comps = root.get()
        for k,v in comps.items():
            if d["cid"] == v["cid"]:
                db.reference("Resolved/").child(d['cid']).set(d)
                root.child(d['cid']).delete()
                break
        return "true"
    except Exception as e:
        print("ERROR! ",e)
        return "false"


'''
@app.route("/resolve_incharge", methods=['POST','GET'])
def sendResult3():
    d = request.get_json()
    # print(request)
    # d={}
    # print(d["image"])
    # d["area"] = "updating through python.."
    # d["numcows"] = "2"
    # d["incharge_id"] = "what is this for?"
    # d["lat"] = "12.123"
    # d["long"] = "80.123"
    # d["fb_uid"] = "incharge id kxznczgo"
    # print(d["date_time"])
    # print(d["lat"])
    # print(d["long"])
    # print(d["numcows"])
    # print(d["fb_uid"])
    print("awsm fam")
    picnp = np.fromstring(base64.b64decode(d["img"]), dtype=np.uint8)
    img = cv2.imdecode(picnp, 1)
    # cv2.imwrite("image2.png",img)
    # dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    # dt_string = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-2]
    # print("datetime string to enter in database:", dt_string)
    # cow_classification_output = getCowClassificationOutput(img)
    cow_classification_output = False
    if not cow_classification_output:
        root = db.reference('Complaints/')
        comps = root.get()
        for k,v in comps.items():
            if d["area"] in v["area"]:
                root2 = db.reference("Resolved/")
                v["resolved_by"] = 'incharge'
                v["resolver"] = d['fb_uid']
                root2.child(v['complaint_id']).set(v)
                root.child(v['complaint_id']).delete()
                break
        return "Resolved the problem. Thank you!"
    else:
        return "There are cows in the image."
'''

if __name__ == "__main__":
    app.run("192.168.43.15",port=5000)
