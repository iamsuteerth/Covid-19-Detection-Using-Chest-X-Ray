# COVID-19 Chest X-Ray Detection Using PyTorch

This project involves building an image classifier to detect COVID-19 in chest X-ray images using deep learning techniques. The dataset consists of chest X-ray images categorized into three classes:
1. **Normal**
2. **Viral Pneumonia**
3. **COVID-19**

The model is trained using a **ResNet-18** architecture, a deep convolutional neural network pre-trained on ImageNet, and fine-tuned for the specific task of classifying chest X-rays.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluation and Results](#evaluation-and-results)
8. [Performance](#performance)
9. [Conclusion](#conclusion)
10. [Acknowledgements](#acknowledgements)

---

## Project Overview

This project aims to classify chest X-ray images into three categories: **Normal**, **Viral Pneumonia**, and **COVID-19**. The model uses the **ResNet-18** deep learning architecture for this classification task. The project is built using **PyTorch**, a popular deep learning framework, and utilizes image pre-processing, data augmentation, and custom data loading techniques to handle the dataset.

---

## Dataset

The dataset used for this project is the **COVID-19 Radiography Database**, which is publicly available on Kaggle. You can find the dataset [here](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

The dataset consists of:
- **Normal** chest X-ray images
- **Viral Pneumonia** chest X-ray images
- **COVID-19** chest X-ray images

### Key Information:
- **Number of images**: ~3000 images
- **Image format**: PNG
- **Classes**: Normal, Viral Pneumonia, COVID-19
- **Class Imbalance**: There are fewer **COVID-19** images compared to **Normal** and **Viral Pneumonia** images.

---

## Dependencies

To run this project, ensure you have the following libraries installed:

```bash
pip install torch torchvision numpy matplotlib Pillow
```

### Required Libraries:
- **PyTorch**: Deep learning framework used for model development and training.
- **Torchvision**: Provides pre-trained models and image transformations.
- **NumPy**: For handling arrays and numerical operations.
- **Matplotlib**: For visualizing images and training results.
- **Pillow**: For loading and manipulating images.

---

## Data Preparation

The dataset is organized into subfolders: `NORMAL`, `Viral Pneumonia`, and `COVID-19`. To split the dataset into training and test sets, a custom script is used to create the required directory structure.

### Steps in Data Preparation:
1. **Rename Directories**: The original folder names are mapped to `normal`, `viral`, and `covid`.
2. **Split into Training and Test Sets**: 30 random images from each class are moved into a separate `test` directory to serve as the test set.
3. **Image Transformations**: Data augmentation and normalization are applied to the training images using **torchvision.transforms**.

```python
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## Model Architecture

For the classification task, **ResNet-18** is used as the base model. This model is pre-trained on the ImageNet dataset and is then fine-tuned for chest X-ray classification. The final layer is modified to output 3 classes (Normal, Viral Pneumonia, COVID-19).

```python
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
```

### Loss Function and Optimizer:
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam` optimizer with a learning rate of 3e-5

---

## Training the Model

The model is trained for 1 epoch, with training and validation loss and accuracy being calculated after each batch. Early stopping is implemented to halt training if the accuracy reaches 95% or higher.

```python
def train(epochs):
    for e in range(epochs):
        # Training loop
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
```

---

## Evaluation and Results

The trained model is evaluated using a validation set. During training, predictions are visualized along with their ground-truth labels to visually inspect the model's performance.

```python
def show_preds():
    resnet18.eval()
    images, labels = next(iter(dl_test))
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds)
```

### Example Visualization:
After training, the model's predictions are displayed alongside the true labels for test images.

---

## Performance

The model achieves **95% accuracy** on the validation set after a few epochs of training. The performance condition was set to halt training once the accuracy exceeds 95%. The model's performance can vary depending on the class distribution and the number of training epochs.

- **Validation Loss**: 0.1131
- **Validation Accuracy**: 0.9778

---

## Conclusion

This project demonstrates how to apply deep learning techniques to classify chest X-ray images into three classes: Normal, Viral Pneumonia, and COVID-19. The use of **ResNet-18** pre-trained on ImageNet significantly helps in achieving good accuracy despite a relatively small dataset. Data augmentation techniques like random horizontal flipping and resizing improve the model’s generalization ability.