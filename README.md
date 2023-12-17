# K-Nearest Neighbors Classifier

## Introduction
This Python script implements a K-Nearest Neighbors (KNN) classifier using the MNIST dataset. The KNN algorithm is applied to classify handwritten digits based on their pixel values. The script also includes methods for calculating distances, finding nearest neighbors, and evaluating the classifier's accuracy.

## Dependencies
Make sure to install the following dependencies before running the script:

```bash
pip install numpy matplotlib scikit-learn
```

##Usage

To run the script, execute the following command:
```bash
python knn_classifier.py
```

## Code Overview

getDistanceForAllTestImage(train, test_image): Calculates distances between test images and all training images.
getLabelsForAllTestImageOrderByDistance(train, train_labels, test_image): Gets labels for test images ordered by distance.
predictImage(train_images, train_labels, query_image, k): Predicts the label for a given test image using KNN.
getAccuracyPerKAndN(distance_array_for_all_test, k, test_labels, n): Calculates accuracy for a specified K and N.
getAccuracyForN(distance_array, test_labels): Calculates accuracy for different values of N.
getAccuracyForFirstK(k, distance_array, test_labels, number_of_test): Calculates accuracy for the first K values.
main(): Main function to run the KNN classifier and visualize results.
## Results

The script prints and plots the accuracy of the KNN classifier for different values of K and N. The accuracy is evaluated based on the MNIST test dataset.