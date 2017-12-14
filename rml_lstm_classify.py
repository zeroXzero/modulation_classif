import numpy as np
import tflearn
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import operator
import cPickle
from numpy import linalg as la 

maxlen = 128 
snrs=""
mods=""
test_idx=""
lbl =""
def gendata(fp, nsamples):
    global snrs, mods, test_idx, lbl
    Xd = cPickle.load(open(fp,'rb'))
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    print mods, snrs
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = X[train_idx]
    X_test =  X[test_idx]
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
   
    return (X_train,X_test,Y_train,Y_test)


def norm_pad_zeros(X_train,nsamples):
    print "Pad:", X_train.shape
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/la.norm(X_train[i,:,0],2)
    return X_train


def to_amp_phase(X_train,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi
    
    
    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))
    
    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1) 
    X_train = np.transpose(np.array(X_train),(0,2,1))
    
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    
    
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train, X_test)


xtrain1,xtest1,ytrain1,ytest1 = gendata("RML2016.10a_dict.dat",128)


xtrain1,xtest1 = to_amp_phase(xtrain1,xtest1,128)

xtrain1 = xtrain1[:,:maxlen,:]
xtest1 = xtest1[:,:maxlen,:]

xtrain1 = norm_pad_zeros(xtrain1,maxlen)
xtest1 = norm_pad_zeros(xtest1,maxlen)


X_train = xtrain1
X_test = xtest1

Y_train = ytrain1
Y_test = ytest1

print("--"*50)
print("Training data:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

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
    alphabet = mods 
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt

class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.accuracy = 0.0

    def on_epoch_end(self, training_state):
        print "accuracy2:", training_state.val_acc 
        if self.accuracy<training_state.val_acc:
           self.accuracy = training_state.val_acc 
           print "Model saved:", self.accuracy 
           self.model.save('lstm_apclaasify_newtf.tfl')


network = tflearn.input_data(shape=[None, maxlen, 2],name="inp")
network = tflearn.lstm(network, 128, return_seq=True, dynamic=True, dropout=(1, 0.8))
network = tf.transpose(tf.stack(network),[1,0,2])
network = tflearn.lstm(network, 128, dynamic=True, dropout=(0.8,1))
network = tflearn.fully_connected(network, len(mods), activation='softmax',name="out")
network = tflearn.regression(network, optimizer='adam',
                 loss='categorical_crossentropy',
                 learning_rate=0.001)
model = tflearn.DNN(network,tensorboard_verbose=0)

monitorCallback = MonitorCallback(model)

Train = False
if Train:
    model.fit(X_train, Y_train, n_epoch=80, shuffle=True,show_metric=True, batch_size=400,validation_set=(X_test, Y_test), run_id='radio_lstm', callbacks=monitorCallback)
else:
    model.load('lstm_apclaasify_newtf.tfl')

classes = mods

acc={}
for snr in snrs:
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    width = 4.1 
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("./images/confmat_"+str(snr)+".pdf")
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1 
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor 
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
print(acc)
