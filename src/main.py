import pickle
from math import sqrt
import numpy as np
import sys
import os, os.path

from skimage.color import rgb2gray
import matplotlib.image as mpimg
import glob
import cv2
from sklearn.metrics import f1_score


class ActivationFunctions:

    def sgm(x, Derivative=False):
        a = 1.0 / (1.0 + np.exp(-x))
        if not Derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            out = a
            return out * (1.0 - out)

    def tanh(x, Derivative=False):
        if not Derivative:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x) ** 2

    def softmax(x, Derivative=False):
        x = x - x.max(axis=1, keepdims=True)
        r = np.exp(x)
        r = r / r.sum(axis=1, keepdims=True)
        if not Derivative:
            return r
        else:
            return r * (1 - r)

    def swish(x, Derivative=False):
        # r = x / (1.0 + np.exp(-x))
        z = x * ActivationFunctions.sgm(x, False)
        if not Derivative:
            return z
        else:
            return z + ActivationFunctions.sgm(x, False) * (1 - z)

    def cost_fun(x, y1, Derivative=False):
        r = -1 * np.mean(y1 * np.log(x), axis=0)
        r1 = (x - y1)
        if not Derivative:
            return r
        else:
            return r1

    def relu(x, Derivative=False):
        r = np.maximum(0, x)
        r1 = 1 * (x > 0)
        if not Derivative:
            return r
        else:
            return r1


class mlp:

    def __init__(self, network_shape, activation):

        self.layer_count = len(network_shape)
        self.weight = [0 for i in range(self.layer_count - 1)]
        self.bias = [0 for i in range(self.layer_count - 1)]
        self.network_shape = network_shape
        self.activa = []
        self.acc = []
        self.micro = []
        self.macro = []
        self.means = []
        self.stdevs = []

        # print(self.layer_count)

        k = 0

        for (current_layer, next_layer) in zip(network_shape[:-1], network_shape[1:]):
            if activation[k] == 'sigmoid' or activation[k] == 'softmax':
                learning = np.sqrt(2) / np.sqrt(current_layer + next_layer)
                self.weight[k] = (np.random.randn(current_layer, next_layer) * learning)
                self.bias[k] = (np.random.randn(1, next_layer)) * learning

            elif activation[k] == 'relu' or activation[k] == 'swish':
                learning = 2 * np.sqrt(1 / (current_layer * next_layer))
                self.weight[k] = (learning * np.random.randn(current_layer, next_layer))
                self.bias[k] = (learning * np.random.randn(1, next_layer))

            elif activation[k] == 'tanh':
                learning = 4 * np.sqrt(2 / (current_layer * next_layer))
                self.weight[k] = (learning * np.random.randn(current_layer, next_layer))
                self.bias[k] = (learning * np.random.randn(1, next_layer))

            self.activa.append(activation[k])
            # print(self.weights_matrix)
            k = k + 1

    def forward_prop(self, inp, y1):
        self.in_act = []
        self.in_input = []
        self.in_input.append(inp)
        self.in_act.append(inp)
        for i in range(self.layer_count - 1):
            result = np.dot(self.in_act[i], self.weight[i]) + self.bias[i]
            self.in_input.append(result)
            if self.activa[i] == 'sigmoid':

                val = ActivationFunctions.sgm(result)
            elif self.activa[i] == 'softmax':
                # print("dddd")
                val = ActivationFunctions.softmax(result)
            elif self.activa[i] == 'tanh':
                # print("ddddff")
                val = ActivationFunctions.tanh(result)
            elif self.activa[i] == 'swish':
                val = ActivationFunctions.swish(result)
            else:

                val = ActivationFunctions.relu(result)
            self.in_act.append(val)
        # print(self.in_act)
        return self.in_act[-1]

    def back_prop(self, org_lbl, learn):
        bias_up = ActivationFunctions.cost_fun(self.in_act[-1], org_lbl, 'True') * ActivationFunctions.softmax(
            self.in_input[-1], 'True')
        weight_up = np.dot(np.array(self.in_act[-2]).T, bias_up) / self.in_act[-2].shape[0]
        val = np.mean(bias_up, axis=0, keepdims=True)
        self.weight[-1] -= learn * weight_up
        self.bias[-1] -= learn * val

        for value in range(self.layer_count - 2, 0, -1):
            bias_up = np.dot(bias_up, np.array(self.weight[value]).T) * (
                ActivationFunctions.sgm(self.in_input[value], 'True'))

            weight_up = np.dot(np.array(self.in_act[value - 1]).T, bias_up) / np.array(self.in_act[value - 1]).shape[0]

            val = np.mean(bias_up, axis=0, keepdims=True)
            self.weight[value - 1] -= learn * weight_up
            self.bias[value - 1] -= learn * val

    def gradient(self, epoch, size, n1):
        out_lbl = np.array(self.label)

        inp = np.array(self.full_input)
        n = len(inp)
        for j in range(epoch):
            #            print(j)
            #            print(np.array(self.full_input).shape)

            # np.random.shuffle(inp)
            i = np.arange(np.array(out_lbl).shape[0])
            np.random.shuffle(i)
            inp, out_lbl = inp[i], out_lbl[i]

            self.batches = [inp[k:k + size] for k in range(0, n, size)]
            self.bat_lbl = [out_lbl[k:k + size] for k in range(0, n, size)]
            self.second(n1)

    def second(self, learn):
        k = 0
        #    print("into second")
        for val in self.batches:
            r = self.forward_prop(val, self.bat_lbl[k])
            self.back_prop(self.bat_lbl[k], learn)
            k += 1

        r = self.forward_prop(self.full_input, self.label)
        z = 1 * (r == r.max(axis=1, keepdims=True))
        print(np.mean((z == self.label).all(axis=1)))

    # self.acc.append(np.mean((z == self.label).all(axis=1)))
    # self.micro.append(f1_score(self.label,z,average='micro'))
    # self.macro.append(f1_score(self.label,z,average='macro'))
    # print(self.label)

    def labeltovector(self, size):
        # rint("labelto")
        list1 = []
        list2 = []
        #        print(np.array(self.label).shape)]
        # list_zero=[0 for i in range(size)]
        for i in range(len(self.label)):
            list1 = [0 for i in range(size)]
            # print("in")
            list1[self.label[i]] = 1
            list2.append(list1)
        # rint("out of for")
        self.label = list2

    #        print(self.label)

    def stand(self):
        a = np.nan_to_num

        self.means, self.stdevs = np.array(self.full_input).mean(axis=0), np.array(self.full_input).std(axis=0)
        self.stdevs = np.where(self.stdevs == 0, 1, self.stdevs)
        self.full_input = a((self.full_input - self.means) / self.stdevs)

    def test_stand(self):
        a = np.nan_to_num

        self.full_input = a((self.full_input - self.means) / self.stdevs)


if __name__ == "__main__":
    arguments = sys.argv

    if arguments[1] == "--test-data":
        test_path = arguments[2]

        if arguments[4] == "MNIST":
            b = open("MNIST_stored_stuff", "rb")
            a = pickle.load(b)
            b.close()

            network_shape = a['list_of_nodes']
            activa = a['activationfunc']

            bpn = mlp(network_shape, activa)
            bpn.weight = a['weights_matrix']
            bpn.bias = a['bias']
            bpn.stdevs = a['std']
            bpn.network_shape = a['list_of_nodes']
            bpn.activa = a['activationfunc']
            bpn.means = a['mean']
            #        bpn.weight= [[np.random.rand(784,30)],[np.random.rand(30,10)]]
            #        bpn.bias=[[np.random.rand(1,30)],[np.random.rand(1,10)]]
            #       bpn.stdevs=np.random.rand(1,784)
            #       bpn.means=np.random.rand(1,784)

            vector = []

            output = []

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/0/*.jpg")]
            images_0 = np.array(images)

            for i in range(len(images_0)):
                output.append(0)

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/1/*.jpg")]
            images_1 = np.array(images)

            for i in range(len(images_1)):
                output.append(1)

            vector = np.vstack((images_0, images_1))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/2/*.jpg")]
            images_2 = np.array(images)

            for i in range(len(images_2)):
                output.append(2)

            vector = np.vstack((vector, images_2))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/3/*.jpg")]
            images_3 = np.array(images)

            for i in range(len(images_3)):
                output.append(3)

            vector = np.vstack((vector, images_3))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/4/*.jpg")]
            images_4 = np.array(images)

            for i in range(len(images_4)):
                output.append(4)

            vector = np.vstack((vector, images_4))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/5/*.jpg")]
            images_5 = np.array(images)

            for i in range(len(images_5)):
                output.append(5)

            vector = np.vstack((vector, images_5))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/6/*.jpg")]
            images_6 = np.array(images)

            for i in range(len(images_6)):
                output.append(6)

            vector = np.vstack((vector, images_6))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/7/*.jpg")]
            images_7 = np.array(images)

            for i in range(len(images_7)):
                output.append(7)

            vector = np.vstack((vector, images_7))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/8/*.jpg")]
            images_8 = np.array(images)

            for i in range(len(images_8)):
                output.append(8)

            vector = np.vstack((vector, images_8))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/9/*.jpg")]
            images_9 = np.array(images)

            for i in range(len(images_9)):
                output.append(9)

            vector = np.vstack((vector, images_9))

            bpn.full_input = np.array(vector)
            bpn.label = np.array(output)

            bpn.labeltovector(10)
            bpn.test_stand()  #############  new function reqd
            # bpn.test_func()
            r = bpn.forward_prop(bpn.full_input, bpn.label)
            #        print("3")
            pr = 1 * (r == r.max(axis=1, keepdims=True))
            #        print("4")

            a = f1_score(np.array(bpn.label), np.array(pr), average='micro')
            b = f1_score(np.array(bpn.label), np.array(pr), average='macro')
            print("F1- Micro: ", end=' ')

            #            print(np.array(bpn.label).shape)
            #            print(np.array(pr))
            print(a)
            print("F1- Macro: ", end=' ')
            acc1 = np.mean((pr == bpn.label).all(axis=1))
            print(b)
            print("Accuracy : ")
            print(acc1 * 100)

        elif arguments[4] == "Cat-Dog":
            b = open("Cat-Dog_stored_stuff", "rb")
            a = pickle.load(b)
            b.close()

            network_shape = a['list_of_nodes']
            activa = a['activationfunc']

            bpn = mlp(network_shape, activa)
            bpn.weight = a['weights_matrix']
            bpn.bias = a['bias']
            bpn.stdevs = a['std']
            bpn.network_shape = a['list_of_nodes']
            bpn.activa = a['activationfunc']
            bpn.means = a['mean']
            #   bpn.weight= []      #dont noe how
            #  bpn.bias=[]           #dont noe how
            #  bpn.stdevs= np.random.rand(1,40000)
            #  bpn.means=np.random.rand(1,40000)

            vector = []
            #            print("1")
            output = []

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/cat/*.jpg")]
            images_0 = np.array(images)

            for i in range(len(images_0)):
                output.append(0)

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/dog/*.jpg")]
            images_1 = np.array(images)

            for i in range(len(images_1)):
                output.append(1)

            vector = np.vstack((images_0, images_1))

            bpn.full_input = np.array(vector)
            bpn.label = np.array(output)
            #           print(np.array(bpn.label).shape)

            bpn.labeltovector(2)
            bpn.test_stand()  ######################
            # bpn.test_func()
            r = bpn.forward_prop(bpn.full_input, bpn.label)
            #        print("3")
            pr = 1 * (r == r.max(axis=1, keepdims=True))
            #        print("4")

            a = f1_score(np.array(bpn.label), np.array(pr), average='micro')
            b = f1_score(np.array(bpn.label), np.array(pr), average='macro')
            print("F1- Micro: ", end=' ')

            #            print(np.array(bpn.label).shape)
            #            print(np.array(pr))
            print(a)
            print("F1- Macro: ", end=' ')
            acc1 = np.mean((pr == bpn.label).all(axis=1))
            print(b)
            print("Accuracy : ")
            print(acc1 * 100)

    elif arguments[1] == "--train-data":

        train_path = arguments[2]
        test_path = arguments[4]
        list1 = arguments[8]
        k = 9
        i = 1
        #       print((list1)[1:])
        while (1):
            if str(arguments[k])[-1] == ']':
                i += 1
                break
            i += 1
            k += 1
        list4 = []
        for i1 in range(i):
            list4.append("sigmoid")
        list4.append("softmax")
        # list1=list1.split(',')
        # list3=arguments[10]
        # list3=list3.split(',')
        #        print(list1)
        #        print(list3)
        if arguments[6] == "MNIST":
            k = 9
            i = 1
            list2 = [784]
            list2.append(int(list1[1:]))
            while (1):
                if arguments[k][-1] == ']':
                    list2.append(int(arguments[k][:-1]))
                    i += 1
                    break
                list2.append(int(arguments[k]))
                i += 1
                k += 1
            list2.append(10)
            # list2=[784]
            # for i in range(len(list1)):
            #     list2.append(int(list1[i]))
            # list2.append(10)

            # list4=[]
            # for i in range(len(list3)):
            #     list4.append(list3[i])
            # list4.append("softmax")
            #           print(list4)
            #           print(list2)
            vector = []

            output = []
            #   print("start loading")
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + '/0/*.jpg')]
            images_0 = np.array(images)

            for i in range(len(images_0)):
                output.append(0)

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/1/*.jpg")]
            images_1 = np.array(images)

            for i in range(len(images_1)):
                output.append(1)
            vector = np.vstack((images_0, images_1))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/2/*.jpg")]
            images_2 = np.array(images)
            for i in range(len(images_2)):
                output.append(2)

            vector = np.vstack((vector, images_2))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/3/*.jpg")]
            images_3 = np.array(images)

            for i in range(len(images_3)):
                output.append(3)

            vector = np.vstack((vector, images_3))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/4/*.jpg")]
            images_4 = np.array(images)

            for i in range(len(images_4)):
                output.append(4)

            vector = np.vstack((vector, images_4))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/5/*.jpg")]
            images_5 = np.array(images)

            for i in range(len(images_5)):
                output.append(5)

            vector = np.vstack((vector, images_5))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/6/*.jpg")]
            images_6 = np.array(images)

            for i in range(len(images_6)):
                output.append(6)

            vector = np.vstack((vector, images_6))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/7/*.jpg")]
            images_7 = np.array(images)

            for i in range(len(images_7)):
                output.append(7)

            vector = np.vstack((vector, images_7))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/8/*.jpg")]
            images_8 = np.array(images)

            for i in range(len(images_8)):
                output.append(8)

            vector = np.vstack((vector, images_8))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(train_path + "/9/*.jpg")]
            images_9 = np.array(images)

            for i in range(len(images_9)):
                output.append(9)

            vector = np.vstack((vector, images_9))
            #   print("end loading")
            #   print(np.array(vector).shape)
            #   print(np.array(output).shape)

            bpn = mlp(list2, list4)
            #   print("end init")
            bpn.label = output
            bpn.full_input = vector
            #      print(np.array(bpn.label).shape)
            # print(bpn.label[2])
            bpn.labeltovector(10)
            #   print(np.array(bpn.label).shape)
            bpn.stand()
            #         print(np.array(bpn.label).shape)
            #            print(bpn.full_input)
            #            print(bpn.label)
            # print("start stand")
            # bpn.stand()
            # print("end stand")
            '''.gradient(500,50,0.01)
            ot=open("MNIST_stored_stuff","wb")
            parameters={}
            parameters['weights_matrix']=bpn.weight
            parameters['bias']=bpn.bias
            parameters['mean']=bpn.means
            parameters['std']=bpn.stdevs
            parameters['list_of_nodes']=bpn.network_shape
            parameters['activationfunc']=bpn.activa
            pickle.dump(parameters,ot)
            ot.close()'''

            #    print("first")
            vector = []

            output = []

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/0/*.jpg")]
            images_0 = np.array(images)

            for i in range(len(images_0)):
                output.append(0)

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/1/*.jpg")]
            images_1 = np.array(images)
            for i in range(len(images_1)):
                output.append(1)

            vector = np.vstack((images_0, images_1))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/2/*.jpg")]
            images_2 = np.array(images)

            for i in range(len(images_2)):
                output.append(2)

            vector = np.vstack((vector, images_2))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/3/*.jpg")]
            images_3 = np.array(images)

            for i in range(len(images_3)):
                output.append(3)

            vector = np.vstack((vector, images_3))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/4/*.jpg")]
            images_4 = np.array(images)

            for i in range(len(images_4)):
                output.append(4)

            vector = np.vstack((vector, images_4))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/5/*.jpg")]
            images_5 = np.array(images)

            for i in range(len(images_5)):
                output.append(5)

            vector = np.vstack((vector, images_5))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/6/*.jpg")]
            images_6 = np.array(images)

            for i in range(len(images_6)):
                output.append(6)

            vector = np.vstack((vector, images_6))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/7/*.jpg")]
            images_7 = np.array(images)

            for i in range(len(images_7)):
                output.append(7)

            vector = np.vstack((vector, images_7))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/8/*.jpg")]
            images_8 = np.array(images)

            for i in range(len(images_8)):
                output.append(8)

            vector = np.vstack((vector, images_8))
            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/9/*.jpg")]
            images_9 = np.array(images)

            for i in range(len(images_9)):
                output.append(9)
            #  print("end of loading tst")
            vector = np.vstack((vector, images_9))
            bpn.full_input = np.array(vector)
            bpn.label = np.array(output)

            bpn.labeltovector(10)
            bpn.test_stand()  ###################
            # print(np.array(bpn.label).shape)
            # bpn.test_func()
            r = bpn.forward_prop(bpn.full_input, bpn.label)
            #        print("3")
            pr = 1 * (r == r.max(axis=1, keepdims=True))
            #        print("4")

            a = f1_score(np.array(bpn.label), np.array(pr), average='micro')
            b = f1_score(np.array(bpn.label), np.array(pr), average='macro')
            print("F1- Micro: ", end=' ')

            #          print(np.array(bpn.label).shape)
            #         print(np.array(pr))
            print(a)
            print("F1- Macro: ", end=' ')
            acc1 = np.mean((pr == bpn.label).all(axis=1))
            print(b)
            print("Accuracy : ")
            print(acc1 * 100)

        elif arguments[6] == "Cat-Dog":
            k = 9
            i = 1
            list2 = [40000]
            list2.append(int(list1[1:]))
            while (1):
                if arguments[k][-1] == ']':
                    list2.append(int(arguments[k][:-1]))
                    i += 1
                    break
                list2.append(int(arguments[k]))
                i += 1
                k += 1
            list2.append(2)
            # list2=[40000]
            # for i in range(len(list1)):
            #     list2.append(int(int(list1[i])))
            # list2.append(2)

            # list4=[]
            # for i in range(len(list3)):
            #     list4.append(list3[i])
            # list4.append("softmax")

            vector = []

            output = []
            # print(train_path)
            images = [cv2.imread(file1, cv2.IMREAD_GRAYSCALE).ravel() for file1 in glob.glob(train_path + '/cat/*.jpg')]
            images_0 = np.array(images)
            # print(images)
            #  print(images_0)
            for i in range(len(images_0)):
                #  print(i)
                output.append(0)

            images = [cv2.imread(file1, cv2.IMREAD_GRAYSCALE).ravel() for file1 in glob.glob(train_path + '/dog/*.jpg')]
            images_1 = np.array(images)

            for i in range(len(images_1)):
                output.append(1)

            vector = np.vstack((images_0, images_1))

            # print("done loading")
            # print(np.array(vector).shape)

            bpn = mlp(list2, list4)
            bpn.label = output
            bpn.full_input = vector
            # print(bpn.label[2])
            bpn.labeltovector(2)
            # print(bpn.full_input)
            # print(bpn.label)
            bpn.stand()
            bpn.gradient(300, 100, 0.01)
            vector = []

            output = []

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/cat/*.jpg")]
            images_0 = np.array(images)

            for i in range(len(images_0)):
                output.append(0)

            images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE).ravel() for file in glob.glob(test_path + "/dog/*.jpg")]
            images_1 = np.array(images)

            for i in range(len(images_1)):
                output.append(1)

            vector = np.vstack((images_0, images_1))

            bpn.full_input = np.array(vector)
            bpn.label = np.array(output)

            bpn.labeltovector(2)
            bpn.test_stand()
            # bpn.test_func()
            r = bpn.forward_prop(bpn.full_input, bpn.label)
            #        print("3")
            pr = 1 * (r == r.max(axis=1, keepdims=True))
            #        print("4")

            a = f1_score(np.array(bpn.label), np.array(pr), average='micro')
            b = f1_score(np.array(bpn.label), np.array(pr), average='macro')
            print("F1- Micro: ", end=' ')

            #            print(np.array(bpn.label).shape)
            #           print(np.array(pr))
            print(a)
            print("F1- Macro: ", end=' ')
            acc1 = np.mean((pr == bpn.label).all(axis=1))
            print(b)
            print("Accuracy : ")
            print(acc1 * 100)
