# AI Image Classifier Web App

This project, undertaken as part of the Even Semester Projects 2024 by the Coding Club of IITG, aims to develop a web application that effectively and accurately distinguishes AI-generated images from real ones. Leveraging open-source Python libraries such as Streamlit and TensorFlow, the application showcases the integration of advanced machine learning techniques and user-friendly web interfaces.

# Dataset Used
<p>
 We have collected a variety of images scraped from Google Search to train the model. Additionally, we have utilized the <b>CIFAKE dataset</b> available on Kaggle to further enhance the model's accuracy.
</p>

# Model Used

<p>
  This repository features a <b>Convolutional Neural Network </b> (CNN) model for binary image classification, developed using TensorFlow and Keras. The architecture comprises four convolutional layers with filter sizes of 32, 64, 128, and 256, each utilizing <b>3x3 kernels</b> and <b>ReLU activation</b> functions to effectively learn and extract intricate image features. Following each convolutional layer, <b>MaxPooling layers</b> with 2x2 windows reduce spatial dimensions and computational load, while <b>Dropout</b> layers with a 0.2 rate prevent overfitting by randomly deactivating 20% of neurons during training. After the convolutional and pooling layers, the model includes a flattening layer and two dense layers with 64 units each, both activated by ReLU, to learn high-level abstractions. The final layer is a single neuron with a <b>sigmoid </b>activation function, designed to output a probability score for binary classification tasks. Compiled with the <b>Adam optimizer</b>, the model leverages <b>binary cross-entropy</b> as the loss function, optimizing for binary classification accuracy. Comprehensive documentation and Jupyter notebooks are provided, detailing data preprocessing, model training, and evaluation processes to facilitate understanding and replication. This robust model is tailored for applications requiring precise binary image classification, ensuring both high performance and ease of use.
</p>

# Installation


Open Git Bash and change the directory to the location where the repository is to be cloned. Then, type the following commands.

```shell
  git init
```
```shell
  git clone https://github.com/chiro2004/AI_Image_Classifier_Web_App.git
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
