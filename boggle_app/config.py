import os
class Config(object):
    MODEL_FN=os.path.join(os.getcwd(),'app','resources','best_2_model.h5')
    IMG_X = 28
    IMG_Y = 28
    BOARD_X = 4
    BOARD_Y = 4
    DICTIONARY_FN = os.path.join(os.getcwd(),'app','resources','dictionary.txt')
    SECRET_KEY = 'you-will-never-guess'
    LABELS_FN = os.path.join(os.getcwd(),'app','resources','labels.npz')
    IMAGES_FN = os.path.join(os.getcwd(),'app','resources','images.npz')
