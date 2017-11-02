#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import pyaudio
import sounddevice as sd
import scipy.io.wavfile as wav
import glob
import numpy as np
import datetime
import pandas as pd
from sklearn.externals import joblib
import os
import xgboost as xgb

from vad import VoiceActivityDetector

from logging import basicConfig, getLogger, ERROR
basicConfig(level=ERROR, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)
error = lambda x: logger.error(x)

class XgbWrapper(object):
    def __init__(self, seed=2017, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 400)

    def fit(self, xtra, ytra):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
    def train(self, xtra, ytra, xte, yte):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        dvalid = xgb.DMatrix(xte, label=yte)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
            watchlist, early_stopping_rounds=10)

    def predict_proba(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

train_wav = glob.glob("train/*.wav")
num_class = len(train_wav)
id_list = list(train_wav)

xgb_params = {}
xgb_params['booster'] = 'gbtree'
xgb_params['objective'] = 'multi:softprob'
xgb_params['num_class'] = num_class
xgb_params['metric'] = 'mlogloss'
xgb_params['learning_rate'] = 0.001
xgb_params['max_depth'] = int(6.0002117448743721)
xgb_params['subsample'] = 0.72476106045336319
xgb_params['min_child_weight'] = int(4.998433055249718)
xgb_params['colsample_bylevel'] = 0.69302144647951536
xgb_params['reg_alpha'] = 0.59125639278096453
xgb_params['gamma'] = 0.11900602913417056

name = "xgb"
model_name = "./model_train.py_xgb.pkl"
clf = joblib.load(model_name)

test_wav = glob.glob("test/*.wav")
def predict_file(f):
    print(f)
    (rate, sig) = wav.read(f)
    return predict_data(rate, sig)
def predict_data(rate, sig):
    pred = {}
    mfcc_feat = mfcc(sig, rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig, rate)
    data = np.c_[mfcc_feat, d_mfcc_feat, fbank_feat]
    Y_pred = clf.predict_proba(data)
    for i in range(Y_pred.shape[1]):
        label = Y_pred[:,i]
        pred[id_list[i]] = np.mean(label)
    return pred


recorded = None
def main():
    fs = 44100
    duration = 10  # seconds
    RATE = 44100

    def callback(indata, outdata, frames, time, status):
        #outdata[:] = indata
        npdata = np.fromstring(indata, dtype=np.int16)
        global recorded
        if recorded is None:
            recorded = npdata
        else:
            recorded = np.r_[recorded, npdata]
        """
        v = VoiceActivityDetector()
        v.data = recorded
        v.data = np.mean(v.data, axis=1, dtype=v.data.dtype)
        v.channels = 1
        v.rate = RATE
        raw_detection = v.detect_speech()
        #raw_detection[:,0] = raw_detection[:,0] / rate * 100
        detected = raw_detection[:,1] == 1
        """
        pred = predict_data(fs, npdata)
        #pred = predict_data(fs, recorded[detected])
        print(pred)

    try:
        with sd.Stream(channels=2, dtype=np.int16, callback=callback):
            sd.sleep(duration * 1000)
            print("#" * 80)
            print("press Return to quit")
            print("#" * 80)
            input()
            print("Stop recording")

            pred = predict_data(fs, recorded)
            print("Predict:")
            print(pred)

            wavfile = 'train/{}.wav'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            wav.write(wavfile, fs, recorded)
            print("Write: " + wavfile)
    except KeyboardInterrupt:
        print("Stop recording")

        pred = predict_data(fs, recorded)
        print("Predict:")
        print(pred)

        wavfile = 'train/{}.wav'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        wav.write(wavfile, fs, recorded)
        print("Write: " + wavfile)

def main2():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 60

# initialize portaudio
    p = pyaudio.PyAudio()
    RATE = int(p.get_device_info_by_index(0)['defaultSampleRate'])
    CHANNELS = int(p.get_device_info_by_index(0)['maxInputChannels'])
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# callback function to stream audio, another thread.


    print("Start recording")
    recorded = None
    statuses = sd.CallbackFlags()
    def callback(data,frame_count, time_info, status):
        global statuses
        statuses |= statuses
        global recorded
        npdata = np.fromstring(data, dtype=np.int16)
        if recorded is None:
            recorded = npdata
        else:
            recorded = np.r_[recorded, npdata]
        pred = predict_data(RATE, recorded)
        error(pred)
        return (data, pyaudio.paContinue)
    stream.start_stream()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("* Killed Process")
        print("Stop recording")

        pred = predict_data(RATE, recorded)
        print("Predict:")
        print(pred)

        wavfile = 'train/{}.wav'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        wav.write(wavfile, RATE, recorded)
        print("Write: " + wavfile)


main()
