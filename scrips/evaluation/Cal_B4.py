from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import argparse

import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


def formula_to_index(formula, dic):
    tmp = []
    for word in formula.strip().split(' '):
        if len(word.strip()) > 0:
            if word.strip() in dic:
                tmp.append(dic[word.strip()])
            else:
                tmp.append(dic['UNK'])
    return tmp


def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')

    parser.add_argument('--formulas', dest='formulas_file_path',
                        type=str, required=True,
                        help='Input formulas.txt path')

    parameters = parser.parse_args()
    return parameters


if __name__ == '__main__':

    parameters = process_args()

    f = open(parameters.formulas_file_path,
             encoding='utf-8').readlines()

    labelIndexDic = {}
    for item_f in f:
        labelIndexDic[item_f.strip().split('\t')[0].strip()] = item_f.strip().split('\t')[1] \
            .strip()

    predictDic = {}
    f2 = open(root_path + '/data/result/predict.txt', encoding='utf-8').readlines()

    for item_f2 in f2:
        index = item_f2.strip().split('\t')[0]
        formula = item_f2.strip().split('\t')[1]
        predictDic[index] = formula

    bleuList = []

    for item_p in tqdm(predictDic):
        predict = predictDic[item_p].strip().split(' ')
        label = labelIndexDic[item_p].strip().split(' ')

        if len(label) >= 4:
            if len(predict) < 4:
                bleuList.append(0)
            else:
                tmpBleu1 = sentence_bleu([label], predict, weights=(0, 0, 0, 1))
                bleuList.append(tmpBleu1)

    print("BLEU-4：")
    print(sum(bleuList) / len(bleuList))
