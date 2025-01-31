# TtBA: Two-third Bridge Approach for Decision-Based Attack

## Overview
TtBA is a methodologie for black-box adversarial attack. This README provides instructions for setting up and running these methods.

## Requirements
- **Python**: 3.11.5
- **Libraries**:
  - PyTorch 2.3.0
  - Torchvision 0.18.0

## Installation
1. **Python Setup**: Ensure that you have the correct version of Python installed. If not, download and install it from [Python's official site](https://www.python.org/downloads/release/python-3115/).

2. **Library Installation**:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0
   ```

## Models
Download the required model files into the `/code/model/` directory using the following links:
- VGG19: [Download](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
- ResNet50: [Download](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)
- Inception V3: [Download](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth)
- Vision Transformer (ViT) B_32: [Download](https://download.pytorch.org/models/vit_b_32-d86f8d99.pth)
- EfficientNet B0: [Download](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth)
- DenseNet161: [Download](https://download.pytorch.org/models/densenet161-8d451a50.pth)

## Dataset Setup
### MNIST #The mnist dataset is available without Download;
Download and prepare the MNIST dataset:
```python
import torchvision
import torchvision.transforms as transforms
test_dataset = torchvision.datasets.MNIST(root='./data/', download=True, train=False, transform=transforms.ToTensor())
```

### CIFAR-10
Download and prepare the CIFAR-10 dataset:
```python
import torchvision
import torchvision.transforms as transforms
test_dataset = torchvision.datasets.CIFAR10(root='./data/', download=True, train=False, transform=transforms.ToTensor())
```

### ImageNet
Download the ImageNet dataset from the following Kaggle link:
[ImageNet Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data)

## Usage
Run TtBA using the following command structure. Specify the dataset and other parameters such as epsilon, the number of images, and budget.

```bash
python main.py --dataset=mnist-cnn --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=5 --beginIMG=0 --budget=10000 --remember=0
python main.py --dataset=cifar10-cnn --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=5 --beginIMG=0 --budget=10000 --remember=0
python main.py --dataset=fashionmnist-cnn --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=5 --beginIMG=0 --budget=10000 --remember=0
```

---
