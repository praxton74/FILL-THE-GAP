# Image Inpainting using a U-Net model with a fused ConvMixer encoder

In this project, I've have tried making use of a fused-encoder architecture with the [U-Net](https://arxiv.org/abs/1505.04597) model for the task of Image Inpainting. The fused-encoder is the recently proposed [ConvMixer](https://openreview.net/forum?id=TVHS5Y4dNvM) architecture <br>

This type of fusion model is inspired by [this project](https://drive.google.com/file/d/1hn9hGkW40AVWv1ZxCaF1Vl86n6d7OyVJ/view) by Ini Oguntola where they used this architecture type for image colorization task. 

![model](asset/fused_model.png)
