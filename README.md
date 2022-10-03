Firstly clone the repo:

`git clone https://github.com/AndrewSultan/ptq_resnet20.git`

than activate venv and install requirements:

`cd ptq_resnet20`

`python3 -m venv .`

`source ./bin/activate`

`pip install -r requirements.txt`

for tests use

`python -m pytest -s ./tests/`

The ResNet20 is defined in ./src/model.py. To train it just run:

`python ./src/train.py`

For validation you can use:

`python ./src/valid.py`.

The model shows quite good results (for resnet of course). 

| metrics   | values |
|-----------|--------|
| accuracy  | 0.9153 |
| precision | 0.9166 |
| recall    | 0.9153 |
| f1        | 0.9155 |



To quantize model with help of pytorch use:

`python ./src/quant_fx.py`

