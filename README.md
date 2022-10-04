<div align="center">
    A Post Training Quantization low-bit library, built on PyTorch, for developing fast and lightweight models for inference.
    <hr/>
</div>


[![CodeFactor](https://www.codefactor.io/repository/github/andrewsultan/ptq_resnet20/badge)](https://www.codefactor.io/repository/github/andrewsultan/ptq_resnet20)

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

Firstly clone the repo:
```bash
git clone https://github.com/AndrewSultan/ptq_resnet20.git
```


then activate venv and install requirements:

```bash
cd ptq_resnet20
python3 -m venv .
source ./bin/activate
pip install -r requirements.txt
```

for tests use

```bash
python -m pytest -s ./tests/
```

The ResNet20 is defined in ./src/model.py. To train it just run:

```bash
python ./src/train.py
```
The best model will be saved in weights folder.

For validation you can use:

```bash
python ./src/valid.py
```

The model shows quite good results (for resnet of course). 

| metrics   | values |
|-----------|--------|
| accuracy  | 0.9153 |
| precision | 0.9166 |
| recall    | 0.9153 |
| f1        | 0.9156 |



To quantize model to 8bit with help of pytorch fx interface use:

```bash
python ./src/quant_fx.py
```

The model will be automatically saved in the qweight folder. Right after that you can simply run:
```bash
python ./src/valid_quant.py
```
Since it finds the last modified model you don't need to specify its name every time.

On validation quantized model shows the following results:

| metrics   | values |
|-----------|--------|
| accuracy  | 0.9155 |
| precision | 0.9160 |
| recall    | 0.9144 |
| f1        | 0.9147 |

The results are surprisingly good, accuracy is even slightly better (perhaps we are just lucky).

OK, let's find out what we can get from more extreme quantization, e.g. 2 and 4 bits.
Also, we are going to check our solution for 8 and 16 bit quantization. 
Since I can't write alone for a week super-duper low-bit GEMM framework we will use 
Fake Quants that will help to imitate low-bit tensor multiplication. 
All tensors will be limited to the maximum possible values for the corresponding number of bits.

But first lets talk about structure of the model that we are going to quantize. 
The model is described in `./src/qmodel.py` file. 
It has the same number of layers as ResNet that we trained earlier.
One of the differences is that it has no BatchNorm.
We got rid of it using BatchNormFolding that is well described in [this article](https://scortex.io/batch-norm-folding-an-easy-way-to-improve-your-network-speed/).
All classes that we use for building our model are in `lbtorch` (low-bit torch) directory.
For convolutions we use class [ConvRelu](https://github.com/AndrewSultan/ptq_resnet20/blob/master/lbtorch/convrelu.py) 
and [LBConv](https://github.com/AndrewSultan/ptq_resnet20/blob/master/lbtorch/convrelu.py#L149). 
It allows us to store the weights of our model in quantized form. 
Thanks to this, we can reduce the state_dict even for a int8 model. 
Class [QFakeStub](https://github.com/AndrewSultan/ptq_resnet20/blob/master/lbtorch/qfakestub.py#L7)
has 2 states: `observe == True` or `observe == False`.
When `True` it just observes min/max values of tensors and calculate 
qparams (scale and zero point) for 2, 4, 8 and 16 bits.
When `False` it quant and then dequant tensor with corresponding quantization.
[LBLinear](https://github.com/AndrewSultan/ptq_resnet20/blob/master/lbtorch/lblinear.py#L11)
is functionally nn.Linear module for low-bits, 
it also can contain weights in quantized form as ConvRelu.
[LBObserver](https://github.com/AndrewSultan/ptq_resnet20/blob/master/lbtorch/lbobserver.py#L6)
is used to quantized weights of convolutions, e.g. quantize per channel, and tensors.
Quantization that do [quant](https://github.com/AndrewSultan/ptq_resnet20/blob/master/lbtorch/functional.py#L59)
is implemented in a naïve way, in the future it is worth trying to implement more 
[advanced](https://arxiv.org/pdf/1909.13144.pdf) schemes. 
The quantization pipeline itself is implemented in `src/quant_lb.py`.


So let's understand step by step how quantization (in this package) works.
1. Firstly, we initialise model that we are going to quantize
and model with pretrained weights.
2. Secondly, we fold (or fuse) model with pretrained weights 
and copy this weights to the other model
3. Then we fit model on several train batches 
so that Fake Observers could collect some statistics
4. Fourth, we quantize model. 
During this step we transfer observers to "fake quant" state,
quantize model weights and save them to the state dict, 
dequantize model weights for inference (since we don't have low-bit GEMM framework).
It should be noted that we **do not** save dequantized weights to state dict.
Therefore, the `state_dict` of int2x4 model can be up to 4 times smaller
than that of int8.
5. And finally we evaluate metrics of resulting model.

The result are presented in the table below. 
Precision, recall, f1 are presented as macro.
They also can be calculated as micro and weighted.

| metrics\\n bits | 2      | 4      | 8      | 16     |
|-----------------|--------|--------|--------|--------|
| Accuracy        | 0.1303 | 0.8600 | 0.9156 | 0.9153 |
| Precision       | 0.1396 | 0.8695 | 0.9167 | 0.9166 |
| Recall          | 0.1303 | 0.8600 | 0.9156 | 0.9153 |
| F1              | 0.1176 | 0.8616 | 0.9159 | 0.9156 |

As we see the table, the quality has dropped significantly on 2 bits.
Most likely this is due to the not the best way of quantizing the weights.

A continuation of this research may be the search of optimal 
quantization for 1/2/4 bit networks 
or for example development of the use of [XNOR approaches](https://arxiv.org/abs/1603.05279)
for network quantization.
