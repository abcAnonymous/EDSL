import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import argparse
import os
import random
from torch.autograd import Variable
from tqdm import tqdm
from torch import optim
from nltk.translate.bleu_score import sentence_bleu
import warnings
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])

sys.path.append(root_path)

from src.model import transformers

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def getBatchIndex(length, batchSize, shuffle=True):
    indexList = list(range(length))
    if shuffle == True:
        random.shuffle(indexList)
    batchList = []
    tmp = []
    for inidex in indexList:
        tmp.append(inidex)
        if len(tmp) == batchSize:
            batchList.append(tmp)
            tmp = []
    if len(tmp) > 0:
        batchList.append(tmp)
    return batchList


def make_tgt_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


####Transformer####
def make_model(tgt_vocab, encoderN=6, decoderN=6,
               d_model=256, d_ff=1024, h=8, dropout=0.0):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = transformers.MultiHeadedAttention(h, d_model, dropout)
    ff = transformers.PositionwiseFeedForward(d_model, d_ff, dropout)
    model = transformers.EncoderDecoder(
        transformers.Encoder(transformers.EncoderLayer(d_model, c(attn), c(ff), dropout),
                             encoderN),
        transformers.Decoder(transformers.DecoderLayer(d_model, c(attn), c(attn),
                                                       c(ff), dropout), decoderN),
        transformers.EncoderPositionalEmbedding(d_model),
        transformers.Embeddings(d_model, tgt_vocab),
        transformers.Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def greedy_decode(model, src, src_position, src_mask, max_len):
    src = F.relu(model.src_proj(src))
    src_position_embed = model.src_position_embed(src_position)

    src_embed = src_position_embed + src

    memory = model.encode2(src_embed, src_mask, src_position_embed)

    lastWord = torch.ones(len(src), 1).cuda().long()
    for i in range(max_len):
        tgt_mask = Variable(subsequent_mask(lastWord.size(1)).type_as(src.data))
        tgt_mask = tgt_mask.repeat(src.size(0), 1, 1)
        out = model.decode(memory, src_mask, Variable(lastWord), tgt_mask)
        prob = model.generator(out[:, -1, :].squeeze(0)).unsqueeze(1)
        _, predictTmp = prob.max(dim=-1)
        lastWord = torch.cat((lastWord, predictTmp), dim=-1)
    prob = model.generator.proj(out)

    return prob


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        return loss


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.mp3 = nn.MaxPool2d(2, 2)
        # self.linear = nn.Linear(576,256)
        # self.cly = nn.Linear(576, label_count)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.mp3(x)

        x_embed = x.view(x.size()[0], -1)

        return x_embed


def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')

    parser.add_argument('--formulas', dest='formulas_file_path',
                        type=str, required=True,
                        help='Input formulas.txt path')

    parser.add_argument('--train', dest='train_file_path',
                        type=str, required=True,
                        help='Input train.txt path')

    parser.add_argument('--val', dest='val_file_path',
                        type=str, required=True,
                        help='Input val.txt path')

    parser.add_argument('--vocab', dest='vocab_file_path',
                        type=str, required=True,
                        help='Input latex_vocab.txt path')

    parameters = parser.parse_args()
    return parameters


if __name__ == '__main__':

    parameters = process_args()


    f = open(parameters.formulas_file_path, encoding='utf-8').readlines()
    labelIndexDic = {}
    for item_f in f:
        labelIndexDic[item_f.strip().split('\t')[0]] = item_f.strip().split('\t')[1]

    f2 = open(parameters.train_file_path, encoding='utf-8').readlines()
    f3 = open(parameters.val_file_path, encoding='utf-8').readlines()

    trainLabelList = []
    for item_f2 in f2:
        if len(item_f2) > 0:
            trainLabelList.append(labelIndexDic[item_f2.strip()])

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

    trainDataArray = np.load(root_path + '/data/preprocess_data/trainData')
    trainPositionArray = np.load(root_path + '/data/preprocess_data/trainPosition')

    trainIndexList = np.load(root_path + '/data/preprocess_data/trainIndexList').tolist()
    trainImgList = np.load(root_path + '/data/preprocess_data/trainImgList')

    labelEmbed_teaching_train_copy = []
    labelEmbed_predict_train_copy = []
    for item_ti in trainIndexList:
        labelEmbed_teaching_train_copy.append(labelEmbed_teaching_train[item_ti])
        labelEmbed_predict_train_copy.append(labelEmbed_predict_train[item_ti])

    labelLenListTrain = []
    for item_pv in labelEmbed_teaching_train_copy:
        count = 0
        for item in item_pv:
            if item != 0:
                count += 1
        labelLenListTrain.append(count)

    trainLabelIndexOrderByLen = np.argsort(np.array(labelLenListTrain)).tolist()

    trainDataArray = trainDataArray[trainLabelIndexOrderByLen].tolist()
    trainPositionArray = trainPositionArray[trainLabelIndexOrderByLen].tolist()

    labelEmbed_teachingArray_train = np.array(labelEmbed_teaching_train_copy)[trainLabelIndexOrderByLen]
    labelEmbed_predictArray_train = np.array(labelEmbed_predict_train_copy)[trainLabelIndexOrderByLen]

    valDataArray = np.load(root_path + '/data/preprocess_data/valData_160')  # .tolist()
    valPositionArray = np.load(root_path + '/data/preprocess_data/valPosition_160')  # .tolist()

    labelLenListVal = []
    for item_pv in labelEmbed_predictArray_val:
        count = 0
        for item in item_pv:
            if item != 0:
                count += 1
        labelLenListVal.append(count)

    valLabelIndexOrderByLen = np.argsort(np.array(labelLenListVal)).tolist()

    valDataArray = valDataArray[valLabelIndexOrderByLen].tolist()
    valPositionArray = valPositionArray[valLabelIndexOrderByLen].tolist()

    labelEmbed_teachingArray_val = labelEmbed_teachingArray_val[valLabelIndexOrderByLen]

    labelEmbed_predictArray_val = labelEmbed_predictArray_val[valLabelIndexOrderByLen]

    BATCH_SIZE = 16

    #####Regularization Parameters####
    dropout = 0.2
    l2 = 1e-4
    #################


    model = make_model(len(index_label_dic) + 3, encoderN=8, decoderN=8,
                       d_model=256, d_ff=1024, dropout=dropout).cuda()
    encoder = Encoder().cuda()

    param = list(model.parameters()) + list(encoder.parameters())

    criterion = LabelSmoothing(size=len(label_index_dic) + 3, padding_idx=0, smoothing=0.1)
    lossComput = SimpleLossCompute(model.generator, criterion)

    learningRate = 3e-4

    totalCount = 0

    exit_count = 0

    bestVal = 0
    bestTrainList = 0

    criterionVal = nn.CrossEntropyLoss(ignore_index=0, size_average=True).cuda()

    while True:

        model.train()
        encoder.train()

        optimizer = optim.Adam(param, lr=learningRate, weight_decay=l2)

        latex_batch_index = getBatchIndex(len(trainIndexList), BATCH_SIZE, shuffle=False)
        random.shuffle(latex_batch_index)

        lossListTrain = []
        latexAccListTrain = []
        bleuListTrain = []

        for batch_i in tqdm(range(len(latex_batch_index))):

            latex_batch = latex_batch_index[batch_i]

            sourceDataTmp = [copy.copy(trainDataArray[item]) for item in latex_batch]
            sourcePositionTmp = [copy.copy(trainPositionArray[item]) for item in latex_batch]
            sourceLengthList = [len(item) for item in sourceDataTmp]
            sourceMaskTmp = [item * [1] for item in sourceLengthList]

            sourceDataTmp = [trainImgList[item] for item in sourceDataTmp]

            sourceLengthMax = max(sourceLengthList)

            for i in range(len(sourceDataTmp)):
                if len(sourceDataTmp[i]) < sourceLengthMax:
                    while len(sourceDataTmp[i]) < sourceLengthMax:
                        sourceDataTmp[i] = np.concatenate((sourceDataTmp[i], np.zeros((1, 30, 30))), axis=0)
                        sourcePositionTmp[i].append([0, 0, 0, 0, 0])
                        sourceMaskTmp[i].append(0)

            sourceDataTmp = np.array(sourceDataTmp)
            sourcePositionTmp = np.array(sourcePositionTmp)
            sourceMaskTmp = np.array(sourceMaskTmp)

            tgt_teaching = labelEmbed_teachingArray_train[latex_batch].tolist()
            tgt_predict = labelEmbed_predictArray_train[latex_batch].tolist()

            tgt_teaching_copy = copy.deepcopy(tgt_teaching)
            tgt_predict_copy = copy.deepcopy(tgt_predict)

            tgtMaxBatch = 0
            for item_tgt in tgt_teaching_copy:
                if len(item_tgt) >= tgtMaxBatch:
                    tgtMaxBatch = len(item_tgt)

            for i in range(len(tgt_teaching_copy)):
                while len(tgt_teaching_copy[i]) < tgtMaxBatch:
                    tgt_teaching_copy[i].append(0)

            for i in range(len(tgt_predict_copy)):
                while len(tgt_predict_copy[i]) < tgtMaxBatch:
                    tgt_predict_copy[i].append(0)

            sourceDataTmpArray = torch.from_numpy(sourceDataTmp).cuda().float()
            sourceMaskTmpArray = torch.from_numpy(sourceMaskTmp).cuda().float().unsqueeze(1)
            sourcePositionTmpArray = torch.from_numpy(sourcePositionTmp).cuda().float()

            tgt_teachingArray = torch.from_numpy(np.array(tgt_teaching_copy)).cuda().float()

            tgt_teachingMask = make_tgt_mask(tgt_teachingArray, 0)

            tgt_predictArray = torch.from_numpy(np.array(tgt_predict_copy)).cuda().long()

            sourceDataTmpArray_input = sourceDataTmpArray.view(-1, 1, 30, 30) / 255

            sourceDataTmpArray = encoder(sourceDataTmpArray_input).view(sourceDataTmpArray.size(0),
                                                                        sourceDataTmpArray.size(1), -1)

            out = model.forward(sourceDataTmpArray, sourcePositionTmpArray, tgt_teachingArray,
                                sourceMaskTmpArray, tgt_teachingMask)

            _, latexPredict = model.generator(out).max(dim=-1)

            for i in range(len(latexPredict)):

                if 2 in tgt_predictArray[i].tolist():
                    endIndex = tgt_predictArray[i].tolist().index(2)
                else:
                    endIndex = MAXLENGTH - 1

                predictTmp = latexPredict[i].tolist()[:endIndex + 1]
                labelTmp = tgt_predictArray[i].tolist()[:endIndex + 1]

                if predictTmp == labelTmp:
                    latexAccListTrain.append(1)
                else:
                    latexAccListTrain.append(0)

                bleuScore = sentence_bleu([labelTmp], predictTmp)
                bleuListTrain.append(bleuScore)

            loss = lossComput(out, tgt_predictArray, len(latex_batch))

            lossListTrain.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trainLoss = sum(lossListTrain) / len(lossListTrain)
        trainAcc = sum(latexAccListTrain) / len(latexAccListTrain)
        trainBleu = sum(bleuListTrain) / len(bleuListTrain)

        model.eval()
        encoder.eval()

        latex_batch_index = getBatchIndex(len(valLabelList), BATCH_SIZE, shuffle=False)

        latexAccListVal = []
        latexLossListVal = []

        with torch.no_grad():
            for batch_i in tqdm(range(len(latex_batch_index))):

                latex_batch = latex_batch_index[batch_i]

                sourceDataTmp = [copy.copy(valDataArray[item]) for item in latex_batch]
                sourcePositionTmp = [copy.copy(valPositionArray[item]) for item in latex_batch]
                sourceLengthList = [len(item) for item in sourceDataTmp]
                sourceMaskTmp = [item * [1] for item in sourceLengthList]

                sourceLengthMax = max(sourceLengthList)

                for i in range(len(sourceDataTmp)):
                    if len(sourceDataTmp[i]) < sourceLengthMax:
                        while len(sourceDataTmp[i]) < sourceLengthMax:
                            sourceDataTmp[i].append(np.zeros((30, 30)))
                            sourcePositionTmp[i].append([0, 0, 0, 0, 0])
                            sourceMaskTmp[i].append(0)

                sourceDataTmp = np.array(sourceDataTmp)
                sourcePositionTmp = np.array(sourcePositionTmp)
                sourceMaskTmp = np.array(sourceMaskTmp)

                tgt_teaching = labelEmbed_teachingArray_val[latex_batch].tolist()
                tgt_predict = labelEmbed_predictArray_val[latex_batch].tolist()

                tgt_teaching_copy = copy.deepcopy(tgt_teaching)
                tgt_predict_copy = copy.deepcopy(tgt_predict)

                tgtMaxBatch = 0
                for item_tgt in tgt_teaching_copy:
                    if len(item_tgt) >= tgtMaxBatch:
                        tgtMaxBatch = len(item_tgt)

                for i in range(len(tgt_teaching_copy)):
                    while len(tgt_teaching_copy[i]) < tgtMaxBatch:
                        tgt_teaching_copy[i].append(0)

                for i in range(len(tgt_predict_copy)):
                    while len(tgt_predict_copy[i]) < tgtMaxBatch:
                        tgt_predict_copy[i].append(0)

                sourceDataTmpArray = torch.from_numpy(sourceDataTmp).cuda().float()
                sourceMaskTmpArray = torch.from_numpy(sourceMaskTmp).cuda().float().unsqueeze(1)
                sourcePositionTmpArray = torch.from_numpy(sourcePositionTmp).cuda().float()

                tgt_teachingArray = torch.from_numpy(np.array(tgt_teaching_copy)).cuda().float()

                tgt_teachingMask = make_tgt_mask(tgt_teachingArray, 0)

                tgt_predictArray = torch.from_numpy(np.array(tgt_predict_copy)).cuda().long()

                sourceDataTmpArray_input = sourceDataTmpArray.view(-1, 1, 30, 30) / 255
                sourceDataTmpArray = encoder(sourceDataTmpArray_input).view(sourceDataTmpArray.size(0),
                                                                            sourceDataTmpArray.size(1), -1)

                out = greedy_decode(model, sourceDataTmpArray, sourcePositionTmpArray, sourceMaskTmpArray, tgtMaxBatch)

                _, latexPredict = out.max(dim=-1)


                for i in range(len(latexPredict)):
                    if 2 in latexPredict[i].tolist():
                        endIndex = latexPredict[i].tolist().index(2)
                    else:
                        endIndex = MAXLENGTH

                    predictTmp = latexPredict[i].tolist()[:endIndex]
                    labelTmp = tgt_predictArray[i].tolist()[:endIndex]

                    if predictTmp == labelTmp:
                        latexAccListVal.append(1)
                    else:
                        latexAccListVal.append(0)

                out = out.contiguous().view(-1, out.size(-1))

                targets = tgt_predictArray.view(-1)
                loss = criterionVal(out, targets)

                latexLossListVal.append(loss.item())

        valAcc = sum(latexAccListVal) / len(latexAccListVal)
        valLoss = sum(latexLossListVal) / len(latexLossListVal)

        if valAcc > bestVal:
            torch.save(model.state_dict(), root_path + '/data/model/model.pkl')
            torch.save(encoder.state_dict(), root_path + '/data/model/encoder.pkl')
            bestVal = valAcc
            exit_count = 0
        else:
            exit_count += 1

            if exit_count > 0 and exit_count % 3 == 0:
                learningRate *= 0.5

        if exit_count == 10:
            exit()

        print("Epoch:" + str(totalCount) + "\t TrainingSet:" + '\t loss:' + str(trainLoss) + "\t ACC:" + str(
            trainAcc) + "\t BLEU:"
              + str(trainBleu) +
              "\t ValSet:" + "\t ACC:" + str(valAcc) + "\t bestAcc:" + str(
            bestVal) + "\t learningRate:"
              + str(learningRate) + "\t exit_count："
              + str(exit_count))

        totalCount += 1
