# AI_Image_Classifier_Web_App

Coding Club Even Semester Project
<h2>Dataset Used</h2>
<p>
 We have collected a variety of images scraped from Google Search to train the model. Additionally, we have utilized the CIFAKE dataset available on Kaggle to further enhance the model's accuracy.
</p>

<h2>Model Used</h2>

<p>
  This repository features a 'Convolutional Neural Network' (CNN) model for binary image classification, developed using TensorFlow and Keras. The architecture comprises four convolutional layers with filter sizes of 32, 64, 128, and 256, each utilizing '3x3 kernels' and 'ReLU activation' functions to effectively learn and extract intricate image features. Following each convolutional layer, 'MaxPooling layers' with 2x2 windows reduce spatial dimensions and computational load, while 'Dropout' layers with a 0.2 rate prevent overfitting by randomly deactivating 20% of neurons during training. After the convolutional and pooling layers, the model includes a flattening layer and two dense layers with 64 units each, both activated by ReLU, to learn high-level abstractions. The final layer is a single neuron with a 'sigmoid' activation function, designed to output a probability score for binary classification tasks. Compiled with the Adam optimizer, the model leverages binary cross-entropy as the loss function, optimizing for binary classification accuracy. Comprehensive documentation and Jupyter notebooks are provided, detailing data preprocessing, model training, and evaluation processes to facilitate understanding and replication. This robust model is tailored for applications requiring precise binary image classification, ensuring both high performance and ease of use.
</p>

# Installation


Open Git Bash and change the directory to the location where the repository is to be cloned. Then, type the following commands.

```shell
  git init
```
```shell
  git clone https://github.com/SanKolisetty/AI-Image-Classifier.git
```
Now, install the requirements using the following command.

```shell
   pip install -r requirements.txt 
```
To access or use the application, open a terminal in the cloned repository folder and run the following command.

```shell
  streamlit run deploy.py
```
Finally, browse the link provided in your browser.
