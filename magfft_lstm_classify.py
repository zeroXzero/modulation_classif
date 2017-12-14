import numpy as np
import tflearn
import numpy as np
import scipy.fftpack as spfft
import tensorflow as tf
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.signal import blackman
import sys
import operator
from collections import OrderedDict
from numpy import linalg as la 
import os


'''
Directory containing dump files with names "<technology-type>_<count>.npy"
e.g. lte_1.npy, lte_2.npy, gsm_1.npy...
This can be easily generated using the labeling tool over the api
'''
mydir="../tech_dumps/"
files = []
for file in os.listdir(mydir):
    if file.endswith(".npy"):
            files.append(os.path.join(mydir, file))

print files
labels={}
lfiles={}
count=0
for f in files:
    fname = f.split("/")[-1]
    if not labels.has_key(fname.split("_")[0]):
        labels[fname.split("_")[0]]=count
        lfiles[fname.split("_")[0]]=[]
        count+=1

    lfiles[fname.split("_")[0]].append(f)


labels = OrderedDict(sorted(labels.items(), key=operator.itemgetter(1)))


print labels
print lfiles 

num_labels = len(labels)
nsamples = 0 

for f in files:
    dta = np.load(f)
    if nsamples < dta.shape[1]:
        nsamples = dta.shape[1]


datatype = "float32"
train_data = np.zeros(nsamples,dtype=datatype)
test_data =  np.zeros(nsamples,dtype=datatype)
train_labels = np.zeros(num_labels)
test_labels = np.zeros(num_labels)


def setup_data():
    global train_data, train_labels, valid_data, valid_labels, test_data, test_labels
    for key in labels.keys():
        print("--"*50)
        for f in lfiles[key]:
            dta = np.load(f)
            res = np.zeros((dta.shape[0],nsamples))
            #append zeros
            res[:,:dta.shape[1]] = dta
            train_cnt = dta.shape[0]/2
            test_cnt = dta.shape[0]/2
            train_data   = np.vstack((train_data,res[0:train_cnt]))
            dummy_labels = np.zeros((train_cnt, len(labels)))
            dummy_labels[:, labels[key]] = 1
            train_labels = np.vstack((train_labels,dummy_labels))
            print("Training data: Generation done for:", key)
            test_data   = np.vstack((test_data,res[train_cnt:train_cnt+test_cnt]))
            dummy_labels = np.zeros((test_cnt, len(labels)))
            dummy_labels[:, labels[key]] = 1
            test_labels = np.vstack((test_labels,dummy_labels))
        print("Testing data: Generation done for:", key)
    train_data = np.delete(train_data,0,0)
    test_data = np.delete(test_data,0,0)
    train_labels = np.delete(train_labels,0,0)
    test_labels = np.delete(test_labels,0,0)


setup_data()
Y_train = train_labels
Y_test  = test_labels


print train_data.shape
print test_data.shape

def lnorm(X_train):
    print "Pad:", X_train.shape
    for i in range(X_train.shape[0]):
        X_train[i,:] = X_train[i,:]/la.norm(X_train[i,:],2)
    return X_train

train_data = lnorm(train_data)
test_data  = lnorm(test_data)

#out0 = (out0-np.mean(out0))/np.std(out0)

X_train = np.reshape(train_data,(-1,nsamples,1))
X_test = np.reshape(test_data,(-1,nsamples,1))

def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    print(cm)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = labels.keys()
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt

class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.accuracy = 0.0

    def on_epoch_end(self, training_state):
        print "accuracy1:", training_state.global_acc 
        print "accuracy2:", training_state.val_acc 
        if self.accuracy<training_state.val_acc:
           self.accuracy = training_state.val_acc 
           print "Model saved:", self.accuracy 
           self.model.save('lstm_tech_classify.tfl')

print("--"*50)
print("Training data:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

network = tflearn.input_data(shape=[None, nsamples, 1],name="inp")
network = tflearn.lstm(network, 128, dynamic=True)
network = tflearn.fully_connected(network, len(labels), activation='softmax',name="out")
network = tflearn.regression(network, optimizer='adam',
                 loss='categorical_crossentropy',
                 learning_rate=0.001)
model = tflearn.DNN(network,tensorboard_verbose=2)

monitorCallback = MonitorCallback(model)

model.fit(X_train, Y_train, n_epoch=8, shuffle=True,show_metric=True, batch_size=100,validation_set=(X_test,Y_test), run_id='lstm_tech', callbacks=monitorCallback)

model.save('../models/lstm_tech_classify.tfl')
classes = labels.keys() 

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



test_X_i = X_test
test_Y_i = Y_test

# estimate classes
test_Y_i_hat = np.array(model.predict(test_X_i))
width = 4.1 
height = width / 1.618
plt.figure(figsize=(width, height))
plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1))
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig("./confmat_tech.pdf")
