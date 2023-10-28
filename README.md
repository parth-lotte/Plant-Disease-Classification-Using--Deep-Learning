
## Plant Disease Classification Using Deep Learning

The deep learning project focused on classifying plant leaf images into three categories. I employed four distinct models - **MobileNetV2, VGG16, ResNet50, and InceptionNetV3 - each trained for 50 epochs.** This effort yielded an initial accuracy rate of 94%.

To further enhance the model's performance, I implemented ensemble stacking techniques, culminating in an impressive 96% accuracy.

To facilitate user interaction, developed an user-friendly webpage  allowing end-users to upload plant images for classification. This application was seamlessly deployed on Microsoft Azure, a leading cloud computing platform.

The technology stack encompassed Python for programming, HTML for web development, TensorFlow for deep learning, Numpy and Scikit-Learn for data processing and evaluation, and Microsoft Azure for robust deployment and hosting. This comprehensive toolset ensured the successful development and deployment of the plant leaf image classification system.
## Dataset

The Dataset consist of three categories : 

**Powdery**

![alt text](https://github.com/parth-lotte/Plant-Disease-Classification-Using--Deep-Learning/blob/main/powdery.jpg)


 **Healthy**
 
 ![alt text](https://github.com/parth-lotte/Plant-Disease-Classification-Using--Deep-Learning/blob/main/Train/Healthy/80a4c7f6ad9a55d8.jpg)
   
 **Rust**

 ![alt text](https://github.com/parth-lotte/Plant-Disease-Classification-Using--Deep-Learning/blob/main/Train/Rust/81b4639b9c72790f.jpg)
## Model Analysis

In this project I used the **Ensemble Stacking** method using the **MobileNetV2, VGG16, ResNet50, and InceptionNetV3** models with an accuracy rate of **94%**. 

**Ensemble stacking**, a meta-ensemble technique in machine learning, combines multiple base models by training a higher-level model to make final predictions. It leverages the diverse strengths of base models to enhance predictive accuracy. Stacking employs techniques like cross-validation to prevent overfitting and can be a powerful tool for improving model performance by aggregating the outputs of various models in a hierarchical manner.


![alt text](https://github.com/parth-lotte/Plant-Disease-Classification-Using--Deep-Learning/blob/main/Ensemble.png)

## Code Flow

The provided code demonstrates a machine learning pipeline for plant disease classification using pre-trained deep learning models.

* **Models and Dependencies:** Load pre-trained deep learning models (MobileNetV2, Inception, ResNet, VGG) using TensorFlow and import necessary libraries.

* **Data Pre-processing:** 

    * Load images from a specified directory using the 'imutils' library.
    * Extract labels from the file paths.
    * Preprocess and normalize images for each model.

* **Label Encoding:**

    * Encode the textual class labels into integer values.
    * Perform one-hot encoding on the integer-encoded labels.

* **Stacking Function (imp_stack):**

    * Generate predictions for each loaded model (m1, m2, m3, m4) on the test dataset.
    * Combine predictions and sum them for the final classification result.
    * Determine the class with the highest score using 'argmax'.

* Model Evaluation:

    * Evaluate the stacked model's performance using the classification_report from      scikit-learn for each model and the stacked model.
    * Report metrics such as precision, recall, and F1-score.

* Additional Data and Evaluation:

    * Load images from a different directory for live data.
    * Preprocess and normalize live data images.
    * Apply the imp_stack function to make predictions on the live data.

* Final Evaluation:

    * Evaluate the stacked model's performance on the live data, providing a classification report.
## Deployment & Tech Stack

The above project was deployed with the help of **Microsoft Azure** and the following tech stack used:

* Python 
* Jinja2
* HTML
* Flask
* CSS
* TensorFlow
* Numpy
* Scikit-Learn

## Authors

- [@parth-lotte](https://www.github.com/parth-lotte)
