import subprocess
from PIL import Image
from tqdm import tqdm

import argparse
from src.utils import ImgCandidate

import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')

    parser.add_argument('--formulas', dest='formulas_file_path',
                        type=str, required=True,
                        help='Input formulas.txt path')

    parameters = parser.parse_args()
    return parameters


if __name__ == '__main__':

    parameters = process_args()

    f = open(root_path + '/data/result/predict.txt').readlines()
    f2 = open(parameters.formulas_file_path, encoding='utf-8').readlines()

    formulaDic = {}
    for item_f2 in f2:
        formulaDic[item_f2.strip().split('\t')[0]] = item_f2.strip().split('\t')[1]

    accList = []

    for item_f in tqdm(f):
        index = item_f.strip().split('\t')[0]
        formula = item_f.strip().split('\t')[1]
        labelFormula = formulaDic[index].strip()

        if formula == labelFormula:
            accList.append(1)
        else:

            pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                      r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + formula + \
                      r'\end{equation*}' + '\n' + '\end{document}'
            f3 = open('predict.tex', mode='w')
            f3.write(pdfText)
            f3.close()
            sub = subprocess.Popen("pdflatex -halt-on-error " + "predict.tex", shell=True, stdout=subprocess.PIPE)
            sub.wait()

            pdfFiles = []
            for _, _, pf in os.walk(os.getcwd()):
                pdfFiles = pf
                break

            if 'predict.pdf' in pdfFiles:
                try:
                    pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                              r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + labelFormula + \
                              r'\end{equation*}' + '\n' + '\end{document}'
                    f3 = open('label.tex', mode='w')
                    f3.write(pdfText)
                    f3.close()
                    sub = subprocess.Popen("pdflatex -halt-on-error " + "label.tex", shell=True, stdout=subprocess.PIPE)
                    sub.wait()

                    os.system(
                        'convert -background white -density 200  -quality 100 -strip ' + 'label.pdf ' + 'label.png')
                    os.system(
                        'convert -background white -density 200  -quality 100 -strip ' + 'predict.pdf ' + 'predict.png')
                    label = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('label.png').convert('L')))).tolist()
                    predict = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('predict.png').convert('L')))).tolist()

                    if label == predict:
                        accList.append(1)
                    else:
                        accList.append(0)
                except:
                    accList.append(0)

            else:
                accList.append(0)
        print(sum(accList) / len(accList))

        os.system('rm -rf *.aux')
        os.system('rm -rf *.log')
        os.system('rm -rf *.tex')
        os.system('rm -rf *.pdf')
        os.system('rm -rf *.png')

    print(sum(accList) / len(accList))
