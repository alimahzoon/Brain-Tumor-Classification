## Brain-Tumor-Classification

Provided by: Ali Mahzoon

---
### Problem
A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.
Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using ConvolutionNeural Network (CNN), Artificial Neural Network (ANN), and TransferLearning (TL) would be helpful to doctors all around the world.

---
### Questions:
1. Which model works best and why?
2. What are the most important evaluation metrics and why?
3. What is the future work to have a better results?

---
### Dataset and Approach
* Dataset and Tools:
   * Given data by Kaggle
   * Tensorflow, Tensorboard, Talos, and Keras library
   * Google colab
   * Neural Networks and Convolutional Neural Networks
   * Pretraind Model (EfficientNetB0)


* Approach:
  * Create a multiclass model to classify Brain Tumors into four classes.
    1.  Glioma
    2.  Meningioma
    3.  No Tumor
    4.  Pituitary


  * Model selected based on the improvement of accuracy in validation dataset.
  * GridSearch used to improve Both NN and CNN models
  * Used holdout set and earlystopping to prevent overfitting
  * Used Tensorboard logs and reduce Learningrate to increase accuracy
  * Data visualized by Tensorboard, Matplotlib and pillow

  ---
### Findings
In this dataset there are 3264 images. As we can see in the following image we have balanced classes. 28.7 percent of images are classified as Glioma, 28.4 percent as Meningioma, 15.3 percent as No Tumor (healthy), and 27.6 percent as Pituitary.

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/1.png "Pre EDA")

The original images size are (512 x 512 x 3), I used 3 different sizes in different models such as (64 x 64 x 3) for Neural Networks, (128 x 128 x 1) for Convolutional Neural Network, and (224 x 224 x 3) for pretrained Convolutional Neural Network. Here we can see four images for each class.

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/2.png " EDA")

For this research I used Talos library to implement Gridsearch on Neural Networks and Convolutional Neural Networks to find the best hyper-parameters for my model.

Best five results for NN are:

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/3.png "NN GridSearch")

Best five results for CNN are:

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/4.png "CNN GridSearch")

The best evaluation metrics by using tensorboard callbacks and early stopping is:

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/5.png "Best Evaluation Metrics")

In this classification "Recall" is the most important metric. For instance, for Glioma tumor, recall metric tells us from all patients with Glioma tumor how many did we label (83%).

In the following photo we can see an Image and the corresponding label bellow it and the plot shows the probability of each tumor being detected as we can see for this particular image the network predicted that there is 49% chance that this image is Meningioma and 17% for each category.

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/6.png "Visualization")

---
### Future work
A CNN can only learn the patterns present in the training data the idea of data augmentation is to change or transform the training data points to artificially create more data, which is a common transformation for images.

A common transformation maybe mirroring the image, so the model will be less biased and less likely to be fooled.

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/7.png "Mirroring")

---
### Conclusion
It is highly likely to overfit validation set during hyper-parameters tuning particularly in image data sets, so having a hold out set that your model has never seen  would be a great asset to check the model performance with unseen dataset.

---
### Recommendations
Using Tensorboard to visualize the histograms of  the model helps to optimize model much faster.
It gives you the ability to check the process of learning In each single neuron in all layers of network.

![Image](https://github.com/alimahzoon/Brain-Tumor-Classification/blob/main/Images/8.png "Tensorboard Histograms")

 
