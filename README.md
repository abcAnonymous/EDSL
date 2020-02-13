# EDSL
 EDSL code
 
## Prerequsites
1. Pytorch
2. Numpy, Pillow
3. Pdflatex
4. ImageMagick

## Dataset Download
ME-98K dataset: https://pan.baidu.com/s/1itQEjPUMve3A3dXezJDedw password: axqh 
ME-20K dataset: https://pan.baidu.com/s/1ti1xfCdj6c36yvy_sHld5Q password: nwks


## Quick Start
### 1. prepocess
`python scripts/preprocessing/preprocess_train.py --formulas data/sample/formulas.txt --train data/sample/train.txt --vocab data/sample/latex_vocab.txt --img data/sample/image_processed/`

`python scripts/preprocessing/preprocess_val.py --formulas data/sample/formulas.txt --val data/sample/train.txt --vocab data/sample/latex_vocab.txt --img data/sample/image_processed/`


`python scripts/preprocessing/preprocess_test.py --formulas data/sample/formulas.txt --test data/sample/test.txt --vocab data/sample/latex_vocab.txt --img data/sample/image_processed/`

### 2. training
`python src/train.py --formulas data/sample/formulas.txt --train data/sample/train.txt  --val data/sample/val.txt --vocab data/sample/latex_vocab.txt`


### 3. testing

`python src/train.py --formulas data/sample/formulas.txt --test data/sample/test.txt  --vocab data/sample/latex_vocab.txt`


### 4. evaluation

`python scripts/evaluation/Cal_B4.py --formulas data/sample/formulas.txt`
`python scripts/evaluation/Cal_R4.py --formulas data/sample/formulas.txt`
`python scripts/evaluation/CalculateMath.py --formulas data/sample/formulas.txt`
`python scripts/evaluation/CalculateMath-WS.py --formulas data/sample/formulas.txt`
