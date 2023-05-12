from django.shortcuts import render
from django.http import HttpResponse
import csv
import io
import pickle
import os
from django.conf import settings
import pickle
import csv
import pandas as pd
import numpy as np

from operator import itemgetter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics


# Create your views here.


def index(request):
    return render(request, 'index.html')


def dataset(request, type):
    return render(request, 'datasets.html', {"type": type})


def feature_check(class_, data__):
    print(data__)


def ml_(class_, csv_file, type_):
    decoded_file = csv_file.read().decode('utf-8').splitlines()
    reader = csv.reader(decoded_file)
    data__ = list(reader)
    data_ = np.array([float(_) for _ in data__[1]]).reshape(1, -1)

    if class_ == "replicated_accoustics":
        feature_check(class_, data__)
        dataset = pd.read_csv(os.path.join(settings.BASE_DIR, 'typo/pickle_files/ReplicatedAcousticFeatures csv.csv'))
        x = dataset.iloc[:, 0:-1]
        y = dataset.iloc[:, -1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size=0.2)
        sc_x = StandardScaler()
        train_x = sc_x.fit_transform(train_x)
        print(data_)
        resp = ""
        with open(os.path.join(settings.BASE_DIR, 'typo/pickle_files/models_replicatedacoustics.pkl'), 'rb') as f:
            data = pickle.load(f)
            for i, model in enumerate(data):
                model_name = f"{model}"
                # print(type(data_set))
                stdc_individual_data = sc_x.transform(data_)
                y_pred = model.predict(stdc_individual_data)
                if type_ == "technician":
                    resp += f"<br/>{model_name}:{y_pred}"
                else:
                    resp = f"Predicted value is {y_pred[0]}"
                    # print(f"{model_name} predictions: {y_pred}")

        return resp
    elif class_ == "spiral_handpd":
        ds = pd.read_csv(os.path.join(settings.BASE_DIR, 'typo/pickle_files/NewSpiral.csv'))
        ds1 = ds.copy()
        ds1.drop(['_ID_EXAM', 'IMAGE_NAME', 'ID_PATIENT', 'GENDER', 'RIGH/LEFT-HANDED', 'AGE'], axis=1, inplace=True)
        temp_cols = ds1.columns.tolist()
        new_cols = temp_cols[1:] + temp_cols[0:1]
        ds1 = ds1[new_cols]
        ds = ds1.dropna()

        x = ds.iloc[:, 0:-1]
        y = ds.iloc[:, -1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size=0.3)

        # scaling
        sc_x = StandardScaler()
        train_x = sc_x.fit_transform(train_x)
        test_x = sc_x.transform(test_x)
        lda = LinearDiscriminantAnalysis(n_components=1)
        train_x_lda = lda.fit_transform(train_x, train_y)
        test_x_lda = lda.transform(test_x)

        with open(os.path.join(settings.BASE_DIR, 'typo/pickle_files/models_spiralpd.pkl'), 'rb') as f:
            data = pickle.load(f)
        stdc_individual_data = sc_x.transform(data_)
        individual_data_lda = lda.transform(stdc_individual_data)

        resp1 = ""
        for i, model in enumerate(data):
            model_name = f"{model}"
            y_pred = model.predict(individual_data_lda)
            if type_ == "technician":
                resp1 += f"<br/>{model_name}:{y_pred}"
            else:
                resp1 = f"Predicted value is {y_pred[0]}"
                # print(f"{model_name} predictions: {y_pred}")

        return resp1

    elif class_ == "gait_swing":
        data1 = pd.read_csv(os.path.join(settings.BASE_DIR, 'typo/pickle_files/Gait_Data___Arm_swing.csv'))
        ds = data1.copy()
        ds.drop(['PATNO','EVENT_ID','INFODT'], axis = 1, inplace = True)
        temp_cols=ds.columns.tolist()
        new_cols=temp_cols[1:] + temp_cols[0:1]
        ds=ds[new_cols]
        ds =ds.dropna()

        x= ds.iloc[:, 0:-1]
        y= ds.iloc[:, -1]
        train_x, test_x , train_y, test_y = train_test_split(x, y, random_state=0, test_size=0.3)

        # scaling
        sc_x = StandardScaler()
        train_x = sc_x.fit_transform(train_x)
        test_x = sc_x.transform(test_x)

        #LDA
        lda = LinearDiscriminantAnalysis(n_components=1)
        train_x_lda = lda.fit_transform(train_x, train_y)
        test_x_lda = lda.transform(test_x)

        with open(os.path.join(settings.BASE_DIR, 'typo/pickle_files/models_gaitarm.pkl'), 'rb') as file:
            data = pickle.load(file)

        stdc_individual_data = sc_x.transform(data_)
        individual_data_lda = lda.transform(stdc_individual_data)

        resp2 = ''
        for i, model in enumerate(data):
            model_name = f"{model}"
            y_pred = model.predict(individual_data_lda)
            if type_ == "technician":
                resp2 += f"<br/>{model_name}:{y_pred}"
            else:
                resp2 = f"Predicted value is {y_pred[0]}"
                # print(f"{model_name} predictions: {y_pred}")

        return resp2

    elif class_ == "meanders_handpd":
        ds = pd.read_csv(os.path.join(settings.BASE_DIR, 'typo/pickle_files/NewMeander.csv'))

        x = ds.iloc[:, 0:-1]
        y = ds.iloc[:, -1]
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size=0.3)

        # scaling
        sc_x = StandardScaler()
        train_x = sc_x.fit_transform(train_x)
        test_x = sc_x.transform(test_x)

        with open(os.path.join(settings.BASE_DIR, 'typo/pickle_files/models_meander.pkl'), 'rb') as f:
            data = pickle.load(f)
        stdc_individual_data = sc_x.transform(data_)

        resp3 = ""
        for i, model in enumerate(data):
            model_name = f"{model}"
            y_pred = model.predict(stdc_individual_data)
            if type_ == "technician":
                resp3 += f"<br/>{model_name}:{y_pred}"
            else:
                resp3 = f"Predicted value is {y_pred[0]}"
                # print(f"{model_name} predictions: {y_pred}")

        return resp3

def upload_csv(request, type_, class_):
    if request.method == 'POST' and request.FILES['csv_file']:
        resp = ml_(class_, request.FILES['csv_file'], type_)
        return HttpResponse(resp)
    else:
        return render(request, 'upload_csv.html')
