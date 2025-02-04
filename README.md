
---

# TtBA: Two-third Bridge Approach for Decision-Based Attack

## Overview
TtBA is a methodology designed for black-box adversarial attacks. This README provides all necessary instructions to set up and run the method on different datasets.

## Requirements

- **Python Version**: 3.11.5
- **Libraries**:
  - PyTorch 2.3.0
  - Torchvision 0.18.0

## Installation

1. **Install Python**: Ensure you have Python 3.11.5 installed. If not, download it from the official [Python website](https://www.python.org/downloads/release/python-3115/).

2. **Install Libraries**: Install the required Python libraries using the following command:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0
   ```

## Models

Download the following pre-trained models and place them in the `/code/model/` directory:

- [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
- [ResNet50](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)
- [Inception V3](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth)
- [Vision Transformer (ViT) B_32](https://download.pytorch.org/models/vit_b_32-d86f8d99.pth)
- [EfficientNet B0](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth)
- [DenseNet161](https://download.pytorch.org/models/densenet161-8d451a50.pth)

## Dataset Setup

### MNIST
The MNIST dataset is available for direct download. Use the following code to prepare it:

```python
import torchvision
import torchvision.transforms as transforms

test_dataset = torchvision.datasets.MNIST(
    root='./data/', download=True, train=False, transform=transforms.ToTensor())
```

### CIFAR-10
To download and prepare the CIFAR-10 dataset, use the following:

```python
import torchvision
import torchvision.transforms as transforms

test_dataset = torchvision.datasets.CIFAR10(
    root='./data/', download=True, train=False, transform=transforms.ToTensor())
```

### ImageNet
Download the ImageNet dataset from the following Kaggle link:
- [ImageNet Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data)

## Usage

To run TtBA, use the following command structure. Customize parameters such as the dataset, epsilon, number of images, and budget. 
mnist-cnn is available without download.

```bash
python main.py --dataset=mnist-cnn --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=5 --beginIMG=0 --budget=10000 --remember=1
python main.py --dataset=cifar10-cnn --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=5 --beginIMG=0 --budget=10000 --remember=1
python main.py --dataset=fashionmnist-cnn --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=5 --beginIMG=0 --budget=10000 --remember=1
```

