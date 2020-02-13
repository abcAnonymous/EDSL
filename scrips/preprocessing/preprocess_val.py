import numpy as np
from tqdm import tqdm
import warnings
from PIL import Image

import argparse
import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

from src.utils import ImgCandidate


def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')

    parser.add_argument('--formulas', dest='formulas_file_path',
                        type=str, required=True,
                        help='Input formulas.txt path')

    parser.add_argument('--val', dest='val_file_path',
                        type=str, required=True,
                        help='Input val.txt path')

    parser.add_argument('--vocab', dest='vocab_file_path',
                        type=str, required=True,
                        help='Input latex_vocab.txt path')

    parser.add_argument('--img', dest='img_path',
                        type=str, required=True,
                        help='Input image path')

    parameters = parser.parse_args()
    return parameters


def get_position_vec(positionList):
    xMax = max([item[1] for item in positionList])
    yMax = max([item[3] for item in positionList])

    finalPosition = []
    for item_g in positionList:
        x_1 = item_g[0]
        x_2 = item_g[1]
        y_1 = item_g[2]
        y_2 = item_g[3]

        tmp = [x_1 / xMax, x_2 / xMax, y_1 / yMax, y_2 / yMax, xMax / yMax]
        finalPosition.append(tmp)
    return finalPosition


if __name__ == '__main__':

    parameters = process_args()

    warnings.filterwarnings('ignore')

    f = open(parameters.formulas_file_path, encoding='utf-8').readlines()
    labelIndexDic = {}
    for item_f in f:
        labelIndexDic[item_f.strip().split('\t')[0]] = item_f.strip().split('\t')[1]

    f3 = open(parameters.val_file_path,
              encoding='utf-8').readlines()

    valLabelList = []
    for item_f3 in f3:
        if len(item_f3) > 0:
            valLabelList.append(labelIndexDic[item_f3.strip()])

    MAXLENGTH = 150

    f5 = open(parameters.vocab_file_path, encoding='utf-8').readlines()

    PAD = 0
    START = 1
    END = 2

    index_label_dic = {}
    label_index_dic = {}

    i = 3
    for item_f5 in f5:
        word = item_f5.strip()
        if len(word) > 0:
            label_index_dic[word] = i
            index_label_dic[i] = word
            i += 1
    label_index_dic['unk'] = i
    index_label_dic[i] = 'unk'
    i += 1

    labelEmbed_teaching_val = []
    labelEmbed_predict_val = []

    for item_l in valLabelList:
        tmp = [1]
        words = item_l.strip().split(' ')
        for item_w in words:
            if len(item_w) > 0:
                if item_w in label_index_dic:
                    tmp.append(label_index_dic[item_w])
                else:
                    tmp.append(label_index_dic['unk'])

        labelEmbed_teaching_val.append(tmp)

        tmp = []
        words = item_l.strip().split(' ')
        for item_w in words:
            if len(item_w) > 0:
                if item_w in label_index_dic:
                    tmp.append(label_index_dic[item_w])
                else:
                    tmp.append(label_index_dic['unk'])

        tmp.append(2)
        labelEmbed_predict_val.append(tmp)

    labelEmbed_teachingArray_val = np.array(labelEmbed_teaching_val)
    labelEmbed_predictArray_val = np.array(labelEmbed_predict_val)

    #
    valData = []
    valPosition = []

    for item_f3 in tqdm(f3):
        img = Image.open(parameters.img_path +
                         item_f3.strip() + ".png").convert('L')
        img = Image.fromarray(ImgCandidate.deletePadding(np.array(img)))
        imgInfo = []
        positionInfo = []
        for t in [160]:
            tmp = ImgCandidate.getAllCandidate(img, t)
            for item_t in tmp:
                if item_t[1] not in positionInfo:
                    imgInfo.append(item_t[0])
                    positionInfo.append(item_t[1])
        positionVec = get_position_vec(positionInfo)
        valData.append(imgInfo)
        valPosition.append(positionVec)

    np.save(root_path + '/data/preprocess_data/valData_160', np.array(valData))
    np.save(root_path + '/data/preprocess_data/valPosition_160', np.array(valPosition))

    #
