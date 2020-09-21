import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, stats
from operator import itemgetter
import pickle as pkl
from itertools import groupby, zip_longest
from collections import defaultdict

def read_im(filename,sigma=5):
    img = cv2.imread(filename,0)
#     if img.shape[0] > img.shape[1]:
#         img = np.transpose(img)
    return ndimage.gaussian_filter(img, sigma=sigma)

def threshold_im(img):
    kernel = np.ones((50,50),np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    ret,th1 = cv2.threshold(img,130,255,cv2.THRESH_BINARY)

    return th1

def threshold_letter(img):
    ret,th1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
    return th1


def plot_im(img):
    plt.imshow(img,'gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

def get_rectangles_in_order(rectangles):
    assert len(rectangles == 16)
    r = rectangles
    r = sorted(r,key=itemgetter(0))
    new_r = []
    for idx in range(0,15,4):
        new_r.extend(sorted(r[idx:idx+4],key=itemgetter(1)))
    return new_r

def is_rectangle_overlap(rect1,rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2

    l1 = (x1,y1)
    r1 = (x1+w1,y1-h1)
    l2 = (x2,y2)
    r2 = (x2+w2, y2-h2)

    if (l1[0] > r2[0] or l2[0] > r1[0]):
#         print("c1")
        return False

    if (l1[1] < r2[1] or l2[1] < r1[1]):
#         print("c2")
        return False

    return True

def get_bounding_rectangles(thresh):
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    extent_fxn = lambda c,w,h: cv2.contourArea(c)/(w*h)
    area_fxn = lambda rect: float(rect[2])*rect[3]
    contourList = []

    # get rid of areas that are above 2 standard deviations from mean
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    sorted_counters = sorted_contours[:min(25,len(sorted_contours))]
    contour_areas = [cv2.contourArea(cnt) for cnt in sorted_contours]
    outlier = abs(stats.zscore(contour_areas))>=2
    sorted_contours = [x for idx,x in enumerate(sorted_contours) if outlier[idx] == False]

    # go through contours, get bounded rectangle and decide whether it can be a tile or not
    for idx,cnt in enumerate(sorted_contours):
        epsilon = 0.05*cv2.arcLength(cnt,True)
        app = cv2.approxPolyDP(cnt,epsilon,True)
        (x,y,w,h) = cv2.boundingRect(app)
        if len(app) >= 4 and extent_fxn(cnt,w,h) > 0.2 and len(contourList) < 16:
            contourList.append((x,y,w,h))

    # sanity check: are there rectangles with an overlap?
    # might be a rectangle within a rectangle, so get rid of it
    keep_rect = [True]*len(contourList)
    for i,rect1 in enumerate(contourList[:]):
        for j,rect2 in enumerate(contourList[:]):
            if i != j and is_rectangle_overlap(rect1,rect2):
                keep_rect[max(i,j)] = False # since contourList is sorted by area, least area is the largest index

    r = [cnt for idx,cnt in enumerate(contourList) if keep_rect[idx] == True]

    # sort by grid order
    r = sorted(r,key=itemgetter(0))
    new_r = []
    for idx in range(0,15,4):
        new_r.extend(sorted(r[idx:idx+4],key=itemgetter(1)))

    # todo: if countourList is less than 16, try to guesstimate the missing coordinates??
    return new_r

def process_rectangle(letter_bb):
    letter_bb = threshold_letter(letter_bb)
#     contours, hier = cv2.findContours(letter_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#     letter_bb = cv2.cvtColor(letter_bb,cv2.COLOR_GRAY2RGB)
#     sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     for c in sorted_contours:
#         print(cv2.contourArea(c))
#     cv2.drawContours(letter_bb,sorted_contours[:],-1,(0,255,0),3)
#     plt.imshow(letter_bb)
#     plt.show()
    return letter_bb

def predict_letters(model,images,key,img_x,img_y):

    images = images.reshape(images.shape[0],img_x,img_y,1)
    images = images.astype('float32')
    images /= 255
    y_pred = model.predict(images)
    y_pred = np.argmax(y_pred, axis=1)
    return [key[y] for y in y_pred]

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def get_all_letters_in_image(fn,model,key,img_x,img_y,board_shape):
    img = read_im(fn)
    mask = threshold_im(img)
    rectangles = get_bounding_rectangles(mask)
    if len(rectangles) != board_shape[0]*board_shape[1]:
        print("Please try again")
        return None, None, None

    img = read_im(fn,0)
    image_data = np.zeros((board_shape[0]*board_shape[1],img_x,img_y),dtype='uint8')
    for idx, (x,y,w,h) in enumerate(rectangles):
        letter = process_rectangle(img[y:y+h,x:x+w])
        image_data[idx] = cv2.resize(letter,(img_x,img_y),interpolation=cv2.INTER_AREA)
    labels = predict_letters(model,image_data,key,img_x,img_y)

    # plt.imshow(img,'gray')
    # plt.title(fn)
    # print(labels)
    # plt.show()
    return list(grouper(4,labels)), labels, image_data
