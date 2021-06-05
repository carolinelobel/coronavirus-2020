"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?

This one is different in that it uses the difference in cases
from day to day, rather than the raw number of cases.
"""

from datetime import date
import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
import matplotlib.pyplot as plt

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 5
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

features = np.concatenate(features, axis=0)
targets = np.concatenate(targets, axis=0)
predictions = {}
dates = confirmed.columns[4:]
#print('dates: ',dates)

for val in np.unique(confirmed["Country/Region"]):
    date_count = [i for i in range(0,len(dates)-1)]
    # test data
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)

    # filter the rest of the data to get rid of the country we are
    # trying to predict
    #get all data for individual country
    mask = targets[:, 1] == val
    tr_features = features[mask]
    tr_targets = targets[mask][:, 1]

    above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
    tr_features = np.diff(tr_features[above_min_cases], axis=-1)
        
    if NORMALIZE:
        tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)

    tr_targets = tr_targets[above_min_cases]
        

    # predict
    try:
        
        cases = np.diff(cases.sum(axis=0, keepdims=True), axis=-1)
        max_index = np.argmax(cases[0])
        new_cases = cases[0][max_index:]
        new_cases = np.ndarray.tolist(new_cases)
        new_cases = [new_cases]
        new_cases = np.array(new_cases)
        #print('new cases.shape: ',new_cases.shape)
        date_count = date_count[max_index:]
        date_count = [date_count]
        date_count = np.array(date_count)
        #print('date_count.shape: ',date_count.shape)
        #print('cases: ',cases)
            
        #LINEAR MODEL
        l_reg = sklearn.linear_model.LinearRegression()
        l_reg.fit(date_count,new_cases)
        print('\n',tr_targets[0],'lreg score: ',l_reg.score(date_count,new_cases))

        #logs = np.log(cases[0].astype(float))
        #print('logs: ',logs)
        #print('len logs: ',len(logs))
        #print('len datecount: ',len(date_count))
        #fit = np.polyfit(date_count,logs,1)
        #print(tr_targets[0],'fit: ',fit)
    except:
        try:
            print('\nskipping ',tr_targets[0])
        except: 
            print('\nskipping unknown')
        
        
        

'''with open('results/polyfit.json', 'w') as f:
    json.dump(predictions, f, indent=4)'''
