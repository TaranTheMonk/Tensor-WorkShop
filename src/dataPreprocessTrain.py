from scipy import misc
import glob
from typing import *
import json
import sys
import os

##get images and save as json files

def getImageByName(path: str) -> List:
    ##read in a image and resize it
    ##resize method is bilinear by default
    outputImage = misc.imread(path)
    outputImage = misc.imresize(outputImage, (64, 64))
    outputImage = outputImage.tolist()
    return outputImage

def getData() -> (Dict[str, List[List[List[int]]]], Dict[str, str]):
    ##Get Train Data
    ##output for each image is a list in the size of 32 * 32 * 3
    x_dict = dict()
    y_dict = dict()
    yLabelPosition = 0
    # for label in ['test_stg1', 'test_stg2']:
    for label in ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']:
        yLabelArray = [0] * 8
        yLabelArray[yLabelPosition] += 1
        nameList = glob.glob(os.path.expanduser('~/Desktop/Projects/Tensor-WorkShop/data/train/%s/*.jpg' % label))
        scale = len(nameList)
        counter = 0
        print('Processing folder: %s' % label)
        for nameString in nameList:
            name = nameString.split('/')[-1]
            x_dict.update({name: getImageByName(nameString)})
            y_dict.update({name: yLabelArray})
            counter += 1
            sys.stdout.write('\r' + 'Processing rate: %s%%  ' % round((counter/scale) * 100, 1))
            sys.stdout.flush()
        yLabelPosition += 1
    return x_dict, y_dict

X, Y = getData()

with open('../data/X-Train-Tensor.json', 'w', encoding='utf-8') as f:
    json.dump(X, f)
f.close()

with open('../data/Y-Train-Tensor.json', 'w', encoding='utf-8') as f:
    json.dump(Y, f)
f.close()

