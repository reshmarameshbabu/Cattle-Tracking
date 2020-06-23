from config import *
from dataset import *
from numpy import zeros, asarray, expand_dims, mean
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from tqdm import tqdm

def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in tqdm(dataset.image_ids):
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

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

cfg = CattlePredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='models/', config=cfg)

print("[INFO] Loading Weights...")
model.load_weights('models/mask_rcnn_cattle_config_0004.h5', by_name=True)

# # evaluate model on training dataset
# train_mAP = evaluate_model(train_set, model, cfg)
# print("[INFO] Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("[INFO] Test mAP: %.3f" % test_mAP)
