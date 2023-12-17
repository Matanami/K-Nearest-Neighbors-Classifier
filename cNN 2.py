import random

from matplotlib import pyplot
from scipy.spatial import distance
from collections import Counter
import numpy as np
import numpy.random


from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]].astype(int)
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]].astype(int)


def getDistanceForAllTestImage(train,test_image):
    distance_array_for_all_test = np.zeros((test.shape[0],train.shape[0]))
    for _ in range(test.shape[0]):
        distance_array_for_all_test[_] = np.sqrt(np.sum((train - test_image[_]) ** 2, axis=1))
    return distance_array_for_all_test

def getLabelsForAllTestImageOrderByDistance(train,train_labels,test_image):
    label_array_for_all_test = np.zeros((test.shape[0],train.shape[0]))
    for _ in range(test.shape[0]):
        label_array_for_all_test[_] = getLabelArrayForImage(train,train_labels,test_image[_])
    return label_array_for_all_test

def getLabelArrayForImage(train_images,train_labels,query_image):
    distance_array = calDis(train_images,query_image.reshape(1, -1))
    indices = np.argsort(distance_array)[0]
    order_neighbors = train_labels[indices]
    return order_neighbors


def dis_between_two_vector(v, u):
    return distance.euclidean(v,u)
    #return numpy.linalg.norm(v-u)

def calDis(train_images, query_image):
    distance_array = np.zeros((1,train_images.shape[0]))
    for i in range(train_images.shape[0]):
        distance_array[0][i] = dis_between_two_vector(train_images[i],query_image[0])
    return distance_array

def accuracy(test_label,images_prid):
    return np.sum(test_label==images_prid)

def predictImage(train_images, train_labels, query_image, k):
    distance_array = calDis(train_images,query_image.reshape(1, -1))
    indices = np.argsort(distance_array)[0][:k]
    k_nearest_neighbors = train_labels[indices]
    most_common = Counter(k_nearest_neighbors).most_common()
    return random.choice(most_common)[0]

def getAccuracyPerKAndN(distance_array_for_all_test,k,test_labels,n=1000):
    images_prid = np.zeros(test_labels.shape[0]).astype(int)
    for i in range(test_labels.shape[0]):
        first_n_train = distance_array_for_all_test[i,:n]
        indices = np.argsort(first_n_train)
        k_nearest_neighbors = train_labels[indices[:k]]
        images_prid[i] = np.bincount(k_nearest_neighbors).argmax()
    return accuracy(test_labels,images_prid)/test_labels.shape[0]

def getAccuracyForN(distance_array,test_labels):
    n_array = np.arange(100,5000,100)
    accuracy_array = np.zeros_like(n_array).astype(float)
    for i,e in enumerate(n_array):
        accuracy_array[i] = getAccuracyPerKAndN(distance_array,1, test_labels,e)
    return accuracy_array

def getAccuracyForFirstK(k,distance_array,test_labels,number_of_test=1000):
    accuracy_array = np.zeros(k)
    for _ in range(k):
        accuracy_array[_] = getAccuracyPerKAndN(distance_array,_+1,test_labels,number_of_test)
    return accuracy_array

def main():
    distance_array = getDistanceForAllTestImage(train,test)
    print(getAccuracyPerKAndN(distance_array,10,test_labels))
    accuracy_array_n = getAccuracyForN(distance_array,test_labels)
    accuracy_array = getAccuracyForFirstK(100,distance_array,test_labels)
    print(np.argmax(accuracy_array))
    # classes = np.random.randint(0,3,100)
    pyplot.plot(np.arange(1,101),accuracy_array,"o")
    figure, axis = pyplot.subplots(2)
    axis[0].plot(range(1,101),accuracy_array,"o",markersize=2)
    axis[1].plot(np.arange(100,5000,100), accuracy_array_n,"o",markersize=2)


    pyplot.savefig("Cross-validation on K")
    pyplot.show()

if __name__ == "__main__":
    main()