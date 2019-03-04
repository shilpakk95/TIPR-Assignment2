import numpy as np
from PIL import Image
import os, os.path
import imageio
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
#from sklearn.metrices import accuracy_score, f1_Score


test_path="C:\\Users\\Slipa\\PycharmProjects\\tipra2kdmsit\\data\\MNIST"

vector=[]


output=[]

images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\0\\*.jpg")]
images_0= np.array(images)

for i in range (len(images_0)):
    output.append(0)

images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\1\\*.jpg")]
images_1 = np.array(images)

for i in range (len(images_1)):
    output.append(1)

vector=np.vstack((images_0,images_1))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\2\\*.jpg")]
images_2 = np.array(images)

for i in range (len(images_2)):
    output.append(2)

print("done")

vector=np.vstack((vector,images_2))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\3\\*.jpg")]
images_3 = np.array(images)

for i in range (len(images_3)):
    output.append(3)

vector=np.vstack((vector,images_3))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\4\\*.jpg")]
images_4 = np.array(images)

for i in range (len(images_4)):
    output.append(4)

vector=np.vstack((vector,images_4))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\5\\*.jpg")]
images_5 = np.array(images)

for i in range (len(images_5)):
    output.append(5)

vector=np.vstack((vector,images_5))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\6\\*.jpg")]
images_6 = np.array(images)

for i in range (len(images_6)):
    output.append(6)

vector=np.vstack((vector,images_6))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\7\\*.jpg")]
images_7 = np.array(images)

print("done")

for i in range (len(images_7)):
    output.append(7)

vector=np.vstack((vector,images_7))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\8\\*.jpg")]
images_8 = np.array(images)

for i in range (len(images_8)):
    output.append(8)

vector=np.vstack((vector,images_8))
images = [cv2.imread(file,cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path +"\\9\\*.jpg")]
images_9 = np.array(images)

for i in range (len(images_9)):
    output.append(9)

vector=np.vstack((vector,images_9))



def labeltovector(output):
    size = 10
    list1=[]
    list2=[]
    list_zero=[0 for i in range(size)]
    for i in range (len(output)):
        list1=[0 for i in range(size)]
        #print(list1)
        #print(output[i])
        list1[output[i]]=1
        #print(list1)
        #break
        list2.append(list1)
    #print(list2)
        #break
    return list2

# L = 1000
# _1,_2 = list(np.random.random((L,2))), list(np.random.random((L,2)))
# X1,X2 = [],[]
# Y1,Y2 = [],[]
# rad = 0.8
# for i in range(L):
#     a,b = _1[i][0],_1[i][1]
#     if a**2+b**2<rad**2:
#         Y1.append([1,0])
#         X1.append(_1[i])
#     elif a**2+b**2>=rad**2:
#         Y1.append([0,1])
#         X1.append(_1[i])
#     a,b = _2[i][0],_2[i][1]
#     if a**2+b**2<rad**2:
#         Y2.append([1,0])
#         X2.append(_2[i])
#     elif a**2+b**2>=rad**2:
#         Y2.append([0,1])
#         X2.append(_2[i])
# X1 = np.array(X1)
# X2 = np.array(X2)
# Y1 = np.array(Y1)
# Y2 = np.array(Y2)

# output=[]
# for i in range (10):
#     for j in range (100):
#         output.append(i)

output=labeltovector(np.array(output))

#print(output)

X1=np.array(vector)
Y1=np.array(output)

indx = np.array(list(range(len(X1))))
np.random.shuffle(indx)
X1 = X1[indx]
Y1 = Y1[indx]
print(X1.shape)
print(Y1.shape)
print(Y1)

model = Sequential()
model.add(Dense(784*2, activation='relu', input_dim=X1.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(Y1.shape[1], activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = Adam(lr=0.001,)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X1, Y1,
          epochs=100,
          batch_size=100)
#score = model.evaluate(X1, Y1, batch_size=40)


#lr=0.0001



list1=[42.39,45.49,62.12,64.98,68.08,70.78,78.82,88.51,92.95,94.36,96.81,97.09,98.25,98.55,98.65,98.83,98.89,98.99,99.08,99.13,99.18,99.21,99.23,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25,99.25]
epochs=[i for i in range(51)]
#plt.plot(epochs,list1, color="black")
plt.plot(epochs,list1)
plt.xlabel('Keras_iteration')
plt.ylabel('Keras_accuracy')
plt.legend(['MNIST_dataset'])
#plt.legend()
plt.savefig(os.getcwd()+'/part_1_task_5.png')
plt.show()

plt.clf()
