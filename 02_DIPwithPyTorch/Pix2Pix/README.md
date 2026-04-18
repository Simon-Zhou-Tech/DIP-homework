# Assignment 2 - Pix2Pix with Fully Convolutional Network

### Implement [Pix2Pix](https://phillipi.github.io/pix2pix/) with [Fully Convolutional Layers](https://arxiv.org/abs/1411.4038)

Fill the [Fully Convolutional Network](FCN_network.py#L3) part of `FCN_network.py`, then train the model on the Facades dataset.

---

## Fill Part

```python
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        output = self.deconv5(x)

        return output
```

---

## Network Design

This model is a basic encoder-decoder fully convolutional network.

### Encoder
- `3 -> 64`
- `64 -> 128`
- `128 -> 256`
- `256 -> 512`
- `512 -> 512`

### Decoder
- `512 -> 512`
- `512 -> 256`
- `256 -> 128`
- `128 -> 64`
- `64 -> 3`

All convolution and deconvolution layers use:
- `kernel_size = 4`
- `stride = 2`
- `padding = 1`

The final layer uses `Tanh()` so that the output range matches the normalized image range `[-1, 1]`.

---

## Training

### Dataset
The model is trained on the **Facades Dataset**.

In the provided dataset pipeline:
- the left half of each image is used as input
- the right half is used as target

### Loss Function
```python
criterion = nn.L1Loss()
```

### Optimizer
```python
optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
```

### Scheduler
```python
StepLR(optimizer, step_size=200, gamma=0.2)
```

---

## Running

For Windows:

```python
python download_facades_dataset.py
python train.py
```

For Linux / macOS:

```python
bash download_facades_dataset.sh
python train.py
```

---

## Current Training Observation

According to the current training results:

- the training results are visually better than the validation results
- the validation loss has decreased to around **0.36**
- after that, the validation loss only changes slightly in the last two decimal places

This indicates that the model has basically converged on the validation set.

At the same time, since the training results are better than the validation results, the model may already show **mild overfitting** or have entered a **performance plateau**.

In addition, the current implementation is only a basic FCN + `L1Loss` framework. It does **not** include:
- discriminator
- adversarial loss
- skip connection / U-Net style structure

So the output images are usually:
- smoother
- blurrier
- weaker in texture details

compared with the full Pix2Pix model.

---

## Results

### Training Results
<img src="train_results/epoch_0/result_1.png" alt="val result" width="800">

<img src="train_results/epoch_100/result_1.png" alt="val result" width="800">

<img src="train_results/epoch_200/result_1.png" alt="val result" width="800">

<img src="train_results/epoch_295/result_1.png" alt="val result" width="800">

### Validation Results
<img src="val_results/epoch_0/result_1.png" alt="val result" width="800">

<img src="val_results/epoch_100/result_1.png" alt="val result" width="800">

<img src="val_results/epoch_200/result_1.png" alt="val result" width="800">

<img src="val_results/epoch_295/result_1.png" alt="val result" width="800">

> You can replace the image paths above with the actual epochs that look best in your experiment.

---

## Analysis

From the current results, the model can already learn the rough mapping between the input image and the target image. The main object layout and color distribution can be restored to some extent.

However, the generated images are still not very sharp, and the details in the validation set are weaker than those in the training set. This is reasonable because:

1. the Facades dataset is relatively small
2. the current model is a simple FCN encoder-decoder
3. the training objective only uses `L1Loss`
4. the implementation is not a full GAN-based Pix2Pix

Therefore, the current result is more like a baseline image-to-image translation result instead of the complete Pix2Pix quality shown in the original paper.

---

## Acknowledgement

> 📋 Thanks for the paper: [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)

> 📋 Thanks for the paper: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
