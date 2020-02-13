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

warnings.filterwarnings('ignore')


def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')

    parser.add_argument('--formulas', dest='formulas_file_path',
                        type=str, required=True,
                        help= 'Input formulas.txt path')

    parser.add_argument('--train', dest='train_file_path',
                        type=str, required=True,
                        help='Input train.txt path')

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


        tmp = [x_1 /xMax , x_2 /xMax , y_1  /yMax, y_2 /yMax, xMax/yMax]
        finalPosition.append(tmp)
    return finalPosition



if __name__ == '__main__':


    parameters = process_args()

    f = open(parameters.formulas_file_path, encoding='utf-8').readlines()
    labelIndexDic = {}
    for item_f in f:
        labelIndexDic[item_f.strip().split('\t')[0]] = item_f.strip().split('\t')[1]

    f2 = open(parameters.train_file_path,
              encoding='utf-8').readlines()


    trainLabelList = []
    for item_f2 in f2:
        if len(item_f2) > 0:
            trainLabelList.append(labelIndexDic[item_f2.strip()])



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

    labelEmbed_teaching_train = [] 

    labelEmbed_predict_train = []  
    for item_l in trainLabelList:
        tmp = [1]
        words = item_l.strip().split(' ')
        for item_w in words:
            if len(item_w) > 0:
                if item_w in label_index_dic:
                    tmp.append(label_index_dic[item_w])
                else:
                     
                    tmp.append(label_index_dic['unk'])

        labelEmbed_teaching_train.append(tmp)

        tmp = []
        words = item_l.strip().split(' ')
        for item_w in words:
            if len(item_w) > 0:
                if item_w in label_index_dic:
                    tmp.append(label_index_dic[item_w])
                else:
                    tmp.append(label_index_dic['unk'])

        tmp.append(2)
        labelEmbed_predict_train.append(tmp)



    imgDic = {}
    imgList = []

    trainData = []
    trainPosition = []
    trainIndexList = []


    for i in tqdm(range(len(f2))):
        item_f2 = f2[i]  ##
        img = Image.open(parameters.img_path +
                         item_f2.strip() + ".png").convert('L')
        img = Image.fromarray(ImgCandidate.deletePadding(np.array(img)))
        for t in [160, 180, 200]:

            tmp = ImgCandidate.getAllCandidate(img, t)
            positionInfo = [item[1] for item in tmp]

            #
            positionVec = get_position_vec(positionInfo)
            #
            if t == 160:
                trainIndexList.append(0 + i)

                trainPosition.append(positionVec)
                imgInfo = []

                for item_t in tmp:
                    if str(item_t[0].tolist()) not in imgDic:
                        imgDic[str(item_t[0].tolist())] = len(imgDic)
                        imgList.append(item_t[0])
                    imgInfo.append(imgDic[str(item_t[0].tolist())])
                trainData.append(imgInfo)

            else:
                if positionVec not in trainPosition:
                    trainIndexList.append(0 + i)

                    trainPosition.append(positionVec)
                    imgInfo = []
                    for item_t in tmp:
                        if str(item_t[0].tolist()) not in imgDic:
                            imgDic[str(item_t[0].tolist())] = len(imgDic)
                            imgList.append(item_t[0])
                        imgInfo.append(imgDic[str(item_t[0].tolist())])
                    trainData.append(imgInfo)



    trainData = np.array(trainData)
    trainPosition = np.array(trainPosition)
    trainIndexList = np.array(trainIndexList)
    imgList = np.array(imgList)

    np.save(root_path + '/data/preprocess_data/trainData', np.array(trainData))
    np.save(root_path + '/data/preprocess_data/trainPosition', np.array(trainPosition))
    np.save(root_path + '/data/preprocess_data/trainIndexList', np.array(trainIndexList))
    np.save(root_path + '/data/preprocess_data/trainImgList',np.array(imgList))

