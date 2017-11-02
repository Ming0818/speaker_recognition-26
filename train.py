#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn.metrics import log_loss
import scipy.io.wavfile as wav
import glob
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import os
import json
import xgboost as xgb
import operator
from sklearn.model_selection import KFold
from matplotlib import pylab as plt
from vad import VoiceActivityDetector

from logging import basicConfig, getLogger, ERROR
basicConfig(level=ERROR, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)

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

# (11999, 26)
def get_data(wavlist):
    X_train = None
    y_train = None
    for i, f in enumerate(wavlist):
        print(f)
        (rate,sig) = wav.read(f)

        v = VoiceActivityDetector(f)
        raw_detection = v.detect_speech()
        #raw_detection[:,0] = raw_detection[:,0] / rate * 100
        detected = raw_detection[:,1] == 1
        print(detected)

        print(sig.shape)
        print(raw_detection.shape)
        print(rate)

        mfcc_feat = mfcc(sig,rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(sig,rate)
        data = np.c_[mfcc_feat, d_mfcc_feat, fbank_feat]
        ids = np.ones(fbank_feat.shape[0])
        ids[:] = i
        if X_train is None:
            X_train = data[detected]
            y_train = ids[detected]
        else:
            X_train = np.r_[X_train, data[detected]]
            y_train = np.r_[y_train, ids[detected]]
    return X_train, y_train
train_wav = glob.glob("train/*.wav")
num_class = len(train_wav)
id_list = list(train_wav)

xgb_params = {}
xgb_params['booster'] = 'gbtree'
xgb_params['objective'] = 'multi:softprob'
xgb_params['num_class'] = num_class
xgb_params['metric'] = 'mlogloss'
xgb_params['learning_rate'] = 0.001
#xgb_params['max_depth'] = int(6.0002117448743721)
xgb_params['max_depth'] = 9
xgb_params['subsample'] = 0.72476106045336319
xgb_params['min_child_weight'] = int(4.998433055249718)
#xgb_params['colsample_bytree'] = 0.97058965304691203
xgb_params['colsample_bylevel'] = 0.69302144647951536
xgb_params['reg_alpha'] = 0.59125639278096453
xgb_params['gamma'] = 0.11900602913417056

name = "xgb"
clf = XgbWrapper(seed=2017, params=xgb_params)
"""

name = "svm"
C = 1.
kernel = 'rbf'
gamma  = 0.0103968337769231
estimator = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
clf = OneVsRestClassifier(estimator)
"""

model_name = "model_{}_{}_{}.pkl".format(os.path.basename(__file__), name, num_class)
if os.path.exists(model_name):
    clf = joblib.load(model_name)
else:
    X, y = get_data(train_wav)
    # (35997, 27)
    features = list(range(X.shape[1]))
    print(X.shape)
    print(X)
    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)
    joblib.dump(clf, model_name)

test_wav = glob.glob("test/*.wav")
def predict(f):
    pred = {}
    print(f)
    (rate,sig) = wav.read(f)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    data = np.c_[mfcc_feat, d_mfcc_feat, fbank_feat]
    Y_pred = clf.predict_proba(data)
    for i in range(Y_pred.shape[1]):
        label = Y_pred[:,i]
        pred[id_list[i]] = np.mean(label)
    return pred

for i, f in enumerate(test_wav):
    pred = predict(f)
    print(pred)
    jsvalue = pd.Series(pred).to_json(orient='values')
    print(jsvalue)

X_test, y_test = get_data(test_wav)
Y_pred = clf.predict_proba(X_test)
print(Y_pred.shape)
score = log_loss(y_train, Y_pred)
print('LogLoss {score}'.format(score=score))


# create a feature map
outfile = open('xgb.fmap', 'w')
i = 0
for feat in range(3):
    outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    i = i + 1
outfile.close()

# plot feature importance
importance = clf.gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 20))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')

