# Q2ATransformer-Improving-Medical-VQA-via-an-Answer-Querying-Decoder

## Requirements

```python
# code is runing on pytorch
  pip install -r requirements.txt --user
```

## Pre-process
2. Download Swin Transformer and bert pretrained models from huggingface model hub: https://huggingface.co/models

   
## training
```shell script
# See the config file for model parameter configuration
    python train.py

```

## Dataset
two public medical VQA datasets: 
PathVQA dataset and download link：
https://arxiv.org/abs/2003.10286
https://vision.aioz.io/f/e0554683595c4e1d9a08/?dl=1

VQA-RAD dataset and download link：
https://www.nature.com/articles/sdata2018251#data-citations
https://vision.aioz.io/f/d6fbe4cef5ac4b948e03/?dl=1
