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
import time
files = ['back_single.jpg',
 'bike.jpg',
 'many.jpg',
 'many_back.jpg',
 'many_test.jpg',
 'single_cow.jpg',
 'single_cow2.jpg',
 'srinaths_cow.jpg',
 'srinaths_cow2.png',
 'street_test.jpg',
 'super_blurry.jpg',
 'test.jpg',
 'test1.jpg']
learn = load_learner('data')
images = []
prediction = []
probability = []
for i in files:
      images.append(i)
      link = i
      img = open_image(link)
      start_time = time.time()
      pred_class,pred_idx,outputs = learn.predict(img)
      print("[INFO] Time taken : ", time.time() - start_time)
      prediction.append(pred_class.obj)
      probability.append(outputs.abs().max().item())
answer = pd.DataFrame({'image_name':images,'label':prediction,'probability':probability})
answer.to_csv("answers.csv",index=False)
