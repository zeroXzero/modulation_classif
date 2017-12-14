#!/usr/bin/env python

import requests
import json
import time,sys

from requests.auth import HTTPBasicAuth
from collections import OrderedDict
from urllib import urlencode

import matplotlib.pyplot as plt
import numpy as np
import optparse
import random 
import getpass
#import initExample ## Add path to library (just for examples; you do not need this)
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
#import tflearn
from numpy import linalg as la 

parser = optparse.OptionParser("usage: %prog -u <username> [-p <password> -r <minfreq,maxfreq> -t <timeresol> -f <frequency_resol>]")
parser.add_option("-u", "--user", dest="username",
                  type="string",
                  help="API username")
parser.add_option("-p", "--pass", dest="password",
                    type="string", help="API password")

parser.add_option("-r", "--range", dest="frange",
                    type="string", help="frequency range separated by commas")

parser.add_option("-t", "--tresol", dest="tresol",
                    type="string", help="time resolution")

parser.add_option("-f", "--fresol", dest="fresol",
                    type="string", help="frequency resolution")

(options, args) = parser.parse_args()
if not options.username:
   parser.error("Username not specified")

if not options.password:
   options.password = getpass.getpass('Password:')

# Electrosense API Credentials 
username=options.username
password=options.password

# Electrosense API
MAIN_URI ='https://test.electrosense.org/api'
SENSOR_LIST = MAIN_URI + '/sensor/list/'
SENSOR_AGGREGATED = MAIN_URI + "/spectrum/aggregated"

r = requests.get(SENSOR_LIST, auth=HTTPBasicAuth(username, password))

if r.status_code != 200:
    print r.content
    exit(-1)

slist_json = json.loads(r.content)

senlist={}
status=[" (off)", " (on)"]

for i, sensor in enumerate(slist_json):
    print "[%d] %s (%d) - Sensing: %s" % (i, sensor['name'], sensor['serial'], sensor['sensing'])
    senlist[sensor['name']+status[int(sensor['sensing'])]]=i

print ""
pos = int( raw_input("Please enter the sensor: "))

print ""
print "   %s (%d) - %s" % (slist_json[pos]['name'], slist_json[pos]['serial'], slist_json[pos]['sensing'])


# Ask for 5 minutes of aggregatd spectrum data

def get_spectrum_data (sensor_id, timeBegin, timeEnd, aggFreq, aggTime, minfreq, maxfreq):
    
    params = OrderedDict([('sensor', sensor_id),
                          ('timeBegin', timeBegin),
                          ('timeEnd', timeEnd),
                          ('freqMin', int(minfreq)),
                          ('freqMax', int(maxfreq)),
                          ('aggFreq', aggFreq),
                          ('aggTime', aggTime),
                          ('aggFun','AVG')])


    r = requests.get(SENSOR_AGGREGATED, auth=HTTPBasicAuth(username, password), params=urlencode(params))

    
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        print "Response: %d" % (r.status_code)
        return None

sp1 = None
sp2 = None
sp3 = None    

epoch_time = int(time.time())
timeBegin = epoch_time - (3600*24*2)
#timeEnd = timeBegin + (3600*20*2)
timeEnd = timeBegin + (60*4)
if not options.fresol:
    freqresol = int(100e3)
else:
    freqresol = int(float(options.fresol))

if not options.tresol:
    tresol = int(60)
else:
    tresol = int(float(options.tresol))

if not options.frange:
    minfreq = 50e6 
    maxfreq = 1500e6
else:
    minfreq = int(float(options.frange.split(",")[0])) 
    maxfreq = int(float(options.frange.split(",")[1])) 

senid = slist_json[pos]['serial'] 
response = get_spectrum_data (slist_json[pos]['serial'], timeBegin, timeEnd, freqresol, tresol, minfreq, maxfreq)
data=np.array(response['values'])
print "Data:",data.shape




# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

pg.mkQApp()
#pg.setConfigOption('background','w')
#pg.setConfigOption('foreground','k')
tab = pg.QtGui.QTabWidget()
tab.show()
grid = QtGui.QGridLayout()
qwid = QtGui.QWidget()
qwid.setLayout(grid)
split1 = QtGui.QSplitter()
grid.addWidget(split1)
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')
scroll= tab.addTab(qwid,"Spectrum")
split1.addWidget(win)

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot()

# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Custom ROI for selecting an image region
bpos = [100,1]
roi = pg.ROI(bpos, [data.shape[1]/10, data.shape[0]/3])
roi.addScaleHandle([0.5, 1], [0.5, 0.5])
roi.addScaleHandle([0, 0.5], [0.5, 0.5])
p1.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image

# Isocurve drawing
iso = pg.IsocurveItem(level=0.8, pen='g')
iso.setParentItem(img)
iso.setZValue(5)

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.gradient.loadPreset("flame")
hist.setImageItem(img)
win.addItem(hist)

# Draggable line for setting isocurve level
isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(isoLine)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
isoLine.setValue(0.8)
isoLine.setZValue(1000) # bring iso line above contrast controls

# Another plot area for displaying ROI data
win.nextRow()
p2 = win.addPlot(colspan=2)
p2.setMaximumHeight(250)
win.resize(800, 800)
save = QtGui.QPushButton('Save')
classif = QtGui.QPushButton('Classify')
fdata = QtGui.QPushButton('Fetch')
cb = pg.ComboBox()
cb.setItems(senlist)
cb.setValue(pos)
stresol = pg.SpinBox(value=tresol, step=1, bounds=[0, None])
sfresol = pg.SpinBox(value=freqresol, step=freqresol, bounds=[0, None])
sminfreq = pg.SpinBox(value=minfreq, step=freqresol, bounds=[0, None])
smaxfreq = pg.SpinBox(value=maxfreq, step=freqresol, bounds=[0, None])
stbegin= pg.SpinBox(value=3600*24*2, step=1, bounds=[0, None])
sduration = pg.SpinBox(value=4, step=1, bounds=[0, None])
slabel= QtGui.QLabel("Sensor")
tlabel= QtGui.QLabel("Time resolution (s)")
flabel= QtGui.QLabel("Frequency resolution (Hz)")
minflabel= QtGui.QLabel("Min freq (Hz)")
maxflabel= QtGui.QLabel("Max freq (Hz)")
dlabel= QtGui.QLabel("Data duration (s)")
tblabel= QtGui.QLabel("Begin time (s): Current time-")
grid2 = QtGui.QGridLayout()
grid3 = QtGui.QGridLayout()
qwid2 = QtGui.QWidget()
qwid3 = QtGui.QWidget()
qwid2.setLayout(grid2)
qwid3.setLayout(grid3)
grid2.addWidget(save,0,0)
grid2.addWidget(classif,0,1)
grid2.addWidget(fdata,0,2)
split1.addWidget(qwid2)
split1.addWidget(qwid3)
grid3.addWidget(slabel,0,0)
grid3.addWidget(cb,0,1)
grid3.addWidget(tlabel,0,2)
grid3.addWidget(stresol,0,3)
grid3.addWidget(flabel,0,4)
grid3.addWidget(sfresol,0,5)
grid3.addWidget(minflabel,1,0)
grid3.addWidget(sminfreq,1,1)
grid3.addWidget(maxflabel,1,2)
grid3.addWidget(smaxfreq,1,3)
grid3.addWidget(dlabel,1,4)
grid3.addWidget(sduration,1,5)
grid3.addWidget(tblabel,1,6)
grid3.addWidget(stbegin,1,7)
split1.setOrientation(0);
win.show()

saveData=""

def savepath():
    global saveData
    fileName = QtGui.QFileDialog.getSaveFileName()
    if fileName:
        outfile = fileName[0]
        np.save(outfile, saveData)
        print "File saved:",outfile

def updsensor(val):
    global slist_json, senid
    senid = slist_json[cb.value()]['serial']

def updtresol(val):
    global tresol
    tresol = int(float(val.value()))

def updfresol(val):
    global freqresol
    freqresol = int(float(val.value()))

def updminfreq(val):
    global minfreq 
    minfreq = int(float(val.value()))

def updmaxfreq(val):
    global maxfreq 
    maxfreq = int(float(val.value()))

def updduration(val):
    global duration, timeBegin, timeEnd 
    duration = int(float(val.value()))
    timeEnd = timeBegin + 60*duration

def updtbegin(val):
    global timeBegin 
    epoch_time = int(time.time())
    timeBegin = epoch_time - int(float(val.value()))

cb.currentIndexChanged.connect(updsensor)
stresol.sigValueChanged.connect(updtresol)
sfresol.sigValueChanged.connect(updfresol)
sminfreq.sigValueChanged.connect(updminfreq)
smaxfreq.sigValueChanged.connect(updmaxfreq)
sduration.sigValueChanged.connect(updduration)
stbegin.sigValueChanged.connect(updtbegin)
save.clicked.connect(savepath)
img.setImage(data)
hist.setLevels(data.min(), data.max())
hist.autoHistogramRange()

# zoom to fit imageo
p1.autoRange()  

text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0; font-size: 16pt;">result</span></div>', anchor=(-0.3,0.5), angle=45, border='w', fill=(0, 0, 255, 100))
#text = pg.TextItem("test", anchor=(0.5, -1.0))

'''
#deeplearn model
nsamples=1024
labels = ["dvb", "radar",  "gsm", "tetra","wfm", "lte"]
network = tflearn.input_data(shape=[None, nsamples, 1],name="inp")
network = tflearn.lstm(network, 128, dynamic=True)
network = tflearn.fully_connected(network, len(labels), activation='softmax',name="out")
network = tflearn.regression(network, optimizer='adam',
                 loss='categorical_crossentropy',
                 learning_rate=0.001)
model = tflearn.DNN(network,tensorboard_verbose=2)
model.load('lstm_tech_classify_gpu.tfl')
'''

def lnorm(X_train):
    print "Pad:", X_train.shape
    for i in range(X_train.shape[0]):
        X_train[i,:] = X_train[i,:]/la.norm(X_train[i,:],2)
    return X_train

'''
def classify():
    global text,nsamples
    #mods = ['OFDM','AM','FM']
    if saveData.shape[1] < nsamples:
        res = np.zeros((saveData.shape[0],nsamples))
        #append zeros
        res[:,:saveData.shape[1]] = saveData 
    else:
        res=saveData[:,:nsamples]
    res = lnorm(res)
    res = np.reshape(res,(-1,nsamples,1))
    pred = np.array(model.predict(res))
    print pred
    print np.argmax(pred, axis=1)
    counts = np.bincount(np.argmax(pred,axis=1))
    mod =  np.argmax(counts)
    ht = '<div style="text-align: center"><span style="color: #FF0; font-size: 12pt;">'+labels[mod]+'</span></div>'
    text = pg.TextItem(html=ht, anchor=(-0.3,0.5), angle=45, border='w', fill=(0, 0, 255, 100))
    updatePlot()
    p2.addItem(text)
'''


def fetch():
    global senid, timeBegin, timeEnd, freqresol, tresol, minfreq, maxfreq, data, img, hist
    #with pg.ProgressDialog("Generating test.hdf5...", 0, 100, cancelText=None, wait=0) as dlg:
    try:
        response = get_spectrum_data(senid, timeBegin, timeEnd, freqresol, tresol, minfreq, maxfreq)
        data=np.array(response['values'])
        hist.setLevels(data.min(), data.max())
        img.setImage(data)
        hist.setImageItem(img)
        print "Data fetched"
        updatePlot()
    except Exception as e:
        print str(e) 


#classif.clicked.connect(classify)
fdata.clicked.connect(fetch)

# Callbacks for handling user interaction
def updatePlot():
    global img, roi, data, p2, saveData, minfreq, freqresol,text
    selected = roi.getArrayRegion(data, img)
    saveData = selected
    print "Selected shape:", np.shape(selected)
    startfreq= minfreq+int(roi.pos()[0]*freqresol)
    stopfreq= startfreq+int(selected.shape[1]*freqresol)
    x = np.arange(startfreq,stopfreq,freqresol)
    xdict = dict(enumerate(x))
    mval = selected.mean(axis=0)
    p2.plot(x,mval, clear=True)
    text.setPos(x[np.argmax(mval)],mval.max())

roi.sigRegionChanged.connect(updatePlot)
updatePlot()

def updateIsocurve():
    global isoLine, iso
    iso.setLevel(isoLine.value())

isoLine.sigDragged.connect(updateIsocurve)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
