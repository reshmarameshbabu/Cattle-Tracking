from dataset import *
import os, time, cv2
from config import *
from mrcnn.model import MaskRCNN
import warnings
warnings.filterwarnings("ignore")


print("[INFO] Prepaaring Train Set...")
train_set = CattleDataset()
train_set.load_dataset("cow-dataset", is_train=True)
train_set.prepare()
print('[INFO] Train: %d...' % len(train_set.image_ids))

print("[INFO]Preparing Test Set...")
test_set = CattleDataset()
test_set.load_dataset('cow-dataset', is_train=False)
test_set.prepare()
print('[INFO] Test: %d' % len(test_set.image_ids))

config = CattleConfig()
config.display()

model = MaskRCNN(mode="training", model_dir="models/", config=config)
print("[INFO] Loading Weights...")
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
print("[INFO] Training model...")
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
