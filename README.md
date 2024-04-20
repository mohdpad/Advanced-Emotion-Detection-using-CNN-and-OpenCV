# Advanced-Emotion-Detection-using-CNN-and-OpenCV

## TABLE OF CONTENTS


1. Abstract	
2. Data Set Description	
3. Data Visualization Techniques	 <br />
a. UMap	 <br />
b. t-SNE	 <br />
c. PCA	
4. Model Architecture <br />	
a. CNN Model	 <br />
b. Experiment 1- CNN	 <br />
c.  Experiment 2 with Early Stopping <br />
d. Experiment 3 Inception with Early Stopping and ReduceLROP	 <br />
5. Additional Features - Real-Time Emotion Detection	 <br />
a. Methodology	 <br />
b. Outputs	
6. Conclusion	
7. Future Work	
8. References	


 <br />

1. Abstract

This project involves the development of a Convolutional Neural Network (CNN) to discern human emotions from facial images. The dataset comprises a substantial 35,887 grayscale images categorized into seven emotional states: happiness, sadness, anger, neutrality, surprise, disgust, and fear. The core innovation of this project lies in real-time emotion detection using OpenCV, coupled with a novel shape design element that provides intuitive visual feedback during the interaction.


![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/f23e4f21-9a96-42f8-8041-7ec794ece2ab)

2. Data Set Description
The dataset utilized in this project consists of 35,887 facial images, with a breakdown of 28,821 images for training and 7,066 for validation. Each image is a 48x48 pixel grayscale representation of various emotional states, including happiness, sadness, anger, neutrality, surprise, disgust, and fear. The dataset is publicly available on Kaggle, accessible through the following:

Dataset- https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/fa402f4c-bdc7-430b-b1e1-619d1e6a8b2a)


The bar chart displays the frequency of various labeled expressions within a given dataset. The x-axis categorizes the data into seven distinct emotional expressions: fear, surprise, sad, happy, neutral, disgust, and angry. The y-axis quantifies the count of occurrences for each expression.  This chart provides a concise overview of the dataset's composition, which can be critical for analysis and model training. 

3. Data Visualization Techniques
a. UMap
Uniform Manifold Approximation and Projection (UMap) is employed to reduce the dimensionality of the dataset while preserving its inherent structure. UMap allows for a more efficient representation of facial expressions, aiding in subsequent model training and analysis. The 2D and 3D UMAP visualizations display a multi-dimensional dataset reduced to fewer dimensions. UMAP effectively simplifies complex data, preserving structure. The 2D plot shows a plane with axes umap1 and umap2, where point density suggests tight clustering and some separation by expressions like 'neutral', 'happy', and 'sad'. The 3D plot adds depth, enhancing data structure analysis and maintaining clustering quality. Such clear groupings benefit classification tasks, indicating that similar items cluster closely. The color-coding highlights distribution patterns, showing that UMAP successfully identifies structure, aiding pattern recognition and informing further data analysis and model development strategies.

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/c6170f58-6061-4ae0-b686-e909de105168)



b. t-SNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is utilized for its ability to visualize high-dimensional data in a two-dimensional or three-dimensional space. This technique helps to reveal patterns and relationships within the facial expression dataset. The images show 2D and 3D t-SNE visualizations of image data, a method that simplifies high-dimensional data while maintaining point distances. The 2D plot reveals a dense point distribution across two axes, with colors indicating classes like 'angry', 'happy', and 'surprise', suggesting various expressions. The 3D plot adds another dimension but still shows intermixed data without clear separation. t-SNE helps understand the data structure, yet the overlap of expressions implies a need for further refinement. These visualizations are foundational for deeper analysis and can direct further steps in data processing and model optimization.

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/7e63fbd2-0421-4e96-85b7-8105036a823b)

c. PCA
Principal Component Analysis (PCA) is applied to further reduce dimensionality and highlight the most significant features contributing to the variability in facial expressions. PCA assists in preprocessing the data before model training. The 2D and 3D PCA visualizations represent a dataset with various expressions labeled in color. PCA, a technique to reduce data dimensions, shows points spread across principal components (PCs). In both visualizations, the points are intermixed, indicating a complex underlying structure with no distinct expression clusters. This suggests that while PCA has reduced data complexity, discerning clear patterns for expressions like 'angry' or 'happy' may require additional analysis or different techniques. These visualizations are critical for identifying nuances in data, guiding further model development, and feature extraction to enhance classification tasks.

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/2dfcbfc2-ba40-49bb-beb1-5d0eb39a9a77)


4. Model Architecture
   
a. Experiment 1 - CNN

Model Architecture:

Input Layer: The network begins with an input layer designed for images of size 
48×48 pixels with a single color channel, typically representing grayscale images.

Convolutional Layers:
The model includes four convolutional layers.
The first layer has 64 filters, the second 128, and the third and fourth layers have 512 filters each. These layers use 3x3 or 5x5 kernels.
Each convolutional layer is followed by batch normalization (to stabilize learning), a ReLU activation function (for non-linearity), a max pooling layer (to reduce the spatial size of the representation, thus reducing the number of parameters and computation in the network), and a dropout layer (with a dropout rate of 0.25 to prevent overfitting).

Fully Connected Layers:
After the convolutional layers, the network includes a flattening step to convert the 2D feature maps into a 1D feature vector.
This is followed by two dense layers with 256 and 512 neurons, respectively. Each dense layer includes batch normalization, a ReLU activation, and a dropout rate of 0.25.

Output Layer: The final layer is a dense layer with 7 neurons (one for each class) and uses a softmax activation function to output a probability distribution over the classes.

Model Compilation:
The model is compiled using the Adam optimizer with a learning rate of 0.0001.
The loss function used is categorical cross-entropy, which is standard for multi-class classification problems.
The model uses accuracy as a metric for performance evaluation.
Training Process:
The model is trained using a generator, allowing efficient handling of large datasets by loading data in batches.
The training is set to run for 48 epochs.
The performance of the model is monitored on a validation dataset.
A ModelCheckpoint callback is used to save the model with the best validation accuracy during training.
Output:

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/9e126b7f-845f-43b6-871e-1636f0b48bc2)


The charts show training progress using Adam optimization. Training loss steadily decreases, which is good. However, validation loss is erratic, suggesting model instability. Similarly, training accuracy improves consistently, but validation accuracy is volatile, indicating the model may not generalize well to unseen data. Further model tuning is advised.

b. Experiment 2 - CNN with Early Stopping

Inclusion of Early Stopping:
This experiment incorporates an Early Stopping callback with specific parameters:
monitor='val_loss': This tells the model to monitor the validation loss for changes.
min_delta=0: This sets the minimum change in the monitored quantity to qualify as an improvement. In this case, any reduction in validation loss is considered an improvement.
patience=3: The training will continue for 3 more epochs even after it stops seeing improvements. If no improvement is seen in the validation loss after these additional epochs, the training process will stop.
verbose=1: Enables verbose output, providing more information about the training process.
restore_best_weights=True: When training is stopped early, the model's weights are rolled back to those of the epoch with the best validation loss.

Impact of Early Stopping:
Efficiency and Prevention of Overfitting: Early Stopping is a form of regularization used to avoid overfitting. The model is prevented from learning noise and irrelevant patterns in the training data by stopping the training when the validation loss stops improving.
Resource Optimization: It saves computational resources by terminating training early if the model ceases to improve, making the training process more efficient.
Model Performance: By reverting to the best weights when early stopping occurs, the model will likely have better generalization performance on unseen data.

Output:

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/c93dd14c-2bea-4d49-a3b6-07f036720ab8)


The graphs show training and validation loss and accuracy for a model using the Adam optimizer. Training loss decreases while validation loss plateaus, suggesting early signs of overfitting. Accuracy improves for both, but validation accuracy lags behind training accuracy, highlighting the need for model calibration to enhance generalization.

c. Experiment 3 - Inception-like Model with Early Stopping and ReduceLROnPlateau

Model Architecture:

Input Layer: The model takes inputs of 48×48 pixel grayscale images.
Parallel Convolutional Towers:
Tower 1: Consists of two convolutional layers with 1×1 and 3×3 kernels.
Tower 2: Similar to Tower 1, but uses a 5×5 kernel for the second layer.
Tower 3: Employs a 3×3 max pooling layer followed by a  1×1 convolution.
Concatenation and Output: The outputs of these towers are concatenated and flattened. The flattened output is then passed through a dense layer with a softmax activation function for classification.
Compilation and Training:
The model is compiled twice: initially without specifying the learning rate, and later with an Adam optimizer set at a learning rate of 0.001.
Categorical cross-entropy is used as the loss function, appropriate for multi-class classification tasks.
The training is conducted over 48 epochs using a generator approach, which is efficient for handling large datasets.
Callbacks Used:
ModelCheckpoint: Saves the best version of the model based on validation accuracy.
EarlyStopping: Monitors the validation loss and stops training if there's no improvement after three epochs. It also restores the weights of the best epoch if early stopping is triggered.
ReduceLROnPlateau: Reduces the learning rate by a factor of 0.2 if the validation loss does not improve, with patience of three epochs. This helps in fine-tuning the model and potentially achieving better performance.

Experiment Goals:
Model's Performance: By using a complex architecture with multiple convolutional towers, the experiment aims to create a model capable of recognizing and classifying a wide range of visual patterns in images.
Efficiency and Optimization: Through callbacks like EarlyStopping and ReduceLROnPlateau, the experiment also focuses on optimizing the training process for efficiency and effectiveness, potentially leading to a model that generalizes well on unseen data.

Output:


![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/24805f1c-c601-4fd1-a178-a2bde687164b)


The graphs depict training and validation loss, and accuracy over epochs, using the Adam optimizer. Loss decreases and accuracy increases as expected, but validation metrics are worse than training, indicating potential overfitting. The accuracy plot suggests the model is learning, but the fluctuating validation accuracy calls for further optimization.

5. Additional Features - Real-Time Emotion Detection
a. Methodology
In addition to the model training and evaluation, real-time emotion detection is implemented using OpenCV. OpenCV's image processing capabilities enable efficient real-time analysis of facial expressions. The shape design element is introduced to provide users with immediate and intuitive visual feedback during the interaction.

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/f47bb002-5b8d-4555-bb59-ed7e76291076)

b. Outputs
Visual feedback is an integral part of the real-time emotion detection process. The shape design element, representing different emotions through distinct shapes, enhances the user's understanding of the detected emotions. The design choices are carefully considered to align with the emotional states, providing a more engaging and informative user experience.

![image](https://github.com/Khizar-Baig/CS584-Final-Project/assets/59732957/b49c8cba-5934-49b7-aa4d-7212d0a17c35)


6. Conclusion

This project successfully combined the power of Convolutional Neural Networks (CNNs) with the real-time processing capabilities of OpenCV to create a system capable of discerning human emotions from facial images. Tackling a dataset of 35,887 grayscale images, the project explored various CNN architectures—from a foundational sequential model to a more complex Inception-like network—each honing in on the subtle nuances of emotions like happiness, sadness, anger, and surprise. The integration of advanced training techniques and a novel shape design element for intuitive user interaction elevated the project, making it not just a technical achievement in AI and machine learning, but also a user-friendly tool for real-time emotion detection. This blend of technical sophistication and practical applicability marks a significant step forward in empathetic technology, opening new avenues for how machines understand and interact with human emotions.

7. Future Work

For future enhancements of this emotion detection project, a multifaceted approach could be adopted. Expanding the dataset to include a wider range of emotional expressions, ages, ethnicities, and lighting conditions would greatly enhance the model's accuracy and inclusivity. Delving into advanced neural architectures like Transformers or enhancing real-time processing capabilities through model optimization techniques would further refine performance. Integrating other modalities such as voice or body language analysis could lead to more comprehensive emotion recognition systems. Additionally, focusing on personalization, ethical usage, and robust privacy measures will be essential, especially as the technology finds applications in sensitive domains like mental health and interactive customer service. Lastly, ensuring the model's adaptability across different platforms would make this technology more accessible and versatile in various real-world scenarios.

8. References <br />

* Emotion Recognition Based on Facial Expressions Using Convolutional Neural Network (CNN) | IEEE Conference Publication | IEEE Xplore. (n.d.). Ieeexplore.ieee.org. Retrieved December 2, 2023, from https://ieeexplore.ieee.org/document/9302866 <br />
* Emotion Detection and Characterization using Facial Features. (n.d.). Ieeexplore.ieee.org. https://ieeexplore.ieee.org/document/8710406  <br />
* Facial Emotion Detection Using Deep Learning. (n.d.). Ieeexplore.ieee.org. https://ieeexplore.ieee.org/document/9154121  <br />
* Facial Emotion Recognition Using Shallow CNN. (n.d.). Springerprofessional.de. Retrieved December 2, 2023, from https://www.springerprofessional.de/en/facial-emotion-recognition-using-shallow-cnn/17867428  <br />
* Huang, Z.-Y., Chiang, C.-C., Chen, J.-H., Chen, Y.-C., Chung, H.-L., Cai, Y.-P., & Hsu, H.-C. (2023). A study on computer vision for facial emotion recognition. 13(1). https://doi.org/10.1038/s41598-023-35446-4 


