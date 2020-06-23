from mrcnn.config import Config

class CattleConfig(Config):
    NAME = "cattle_config"
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 152
    VALIDATION_STEPS = 57

class CattlePredictionConfig(Config):
    NAME = "cattle_config"
    NUM_CLASSES = 1 + 1
    IMAGES_PER_GPU = 1
