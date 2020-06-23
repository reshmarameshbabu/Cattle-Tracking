from config import *
from dataset import *
from numpy import zeros, asarray, expand_dims, mean
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.1)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	pyplot.show()

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

# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)
