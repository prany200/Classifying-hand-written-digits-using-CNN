# Classifying-hand-written-digits-using-CNN
Handwritten Digit Recognition using Convolutional Neural Networks (CNN)

This project aims to develop a Convolutional Neural Network (CNN) model for recognizing handwritten digits from the MNIST dataset.

Overview:

Handwritten digit recognition is a classic problem in the field of machine learning and computer vision, with applications ranging from automated form processing to digitized text recognition. This project presents a CNN-based solution to this problem, leveraging the widely-used MNIST dataset. The code implements the entire pipeline, from data preprocessing to model training and evaluation, providing a comprehensive approach to digit recognition.

Features:

Data Preprocessing: The code preprocesses the MNIST dataset by normalizing pixel values to the range [0, 1] and reshaping the images to fit the CNN model's input shape. Additionally, it performs data augmentation techniques such as rotation and scaling to improve model robustness.

Model Architecture: The CNN architecture consists of multiple convolutional layers followed by max-pooling layers for feature extraction. Batch normalization and dropout layers are incorporated to enhance model generalization and prevent overfitting. The model architecture is highly customizable, allowing users to experiment with different layer configurations and hyperparameters.

Training: The model is trained using the Adam optimizer with categorical cross-entropy loss. Training is performed on the training dataset for a specified number of epochs. The code supports early stopping based on validation loss to prevent overfitting.

Evaluation: After training, the model is evaluated on the test dataset to assess its performance in accurately recognizing handwritten digits. Metrics such as accuracy, precision, recall, and F1 score are computed for evaluation. Additionally, visualizations such as confusion matrices and precision-recall curves are generated to provide insights into the model's performance.

Results:

Upon running the script, the model will be trained and evaluated on the MNIST dataset. The results, including accuracy metrics, performance insights, and visualizations, will be displayed in the console and saved to files for further analysis.
