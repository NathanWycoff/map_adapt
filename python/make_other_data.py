#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  form_kin40.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.11.2024

import numpy as np
import pandas as pd
import os
import shutil
import pickle
import requests, zipfile, io
from ucimlrepo import fetch_ucirepo

exec(open("python/sim_settings.py").read())

staging_dir = 'staging/'

if not os.path.exists(staging_dir):
    os.makedirs(staging_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#for ds in ['kin40k','keggu','year','wine-red','wine-white']:
for ds in datasets_to_use:
    if ds=='kin40k':
        test_X_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_test_data.asc"
        test_y_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_test_labels.asc"
        train_X_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_train_data.asc"
        train_y_url = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_train_labels.asc"

        test_X = np.array(pd.read_csv(test_X_url, sep = "\s+", header = None))
        test_y = np.array(pd.read_csv(test_y_url, sep = "\s+", header = None)).flatten()
        train_X = np.array(pd.read_csv(train_X_url, sep = "\s+", header = None))
        train_y = np.array(pd.read_csv(train_y_url, sep = "\s+", header = None)).flatten()

        X = np.concatenate([train_X, test_X])
        y = np.concatenate([train_y, test_y])
    elif ds=='keggu':
        zip_file_url = 'https://archive.ics.uci.edu/static/public/221/kegg+metabolic+reaction+network+undirected.zip'
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(staging_dir)

        datstr = 'Reaction Network (Undirected).data'
        df = pd.read_csv(staging_dir+datstr, header = None)
        # Fill missing with zero in this benchmark.
        df = df.iloc[:,1:].apply(lambda x: pd.to_numeric(x, errors = 'coerce'), axis = 0).fillna(0)

        X = np.array(df.iloc[:,1:]).astype(float)
        y = np.log10(np.array(df.iloc[:,0]).astype(float))
    elif ds=='year':
        zip_file_url = 'https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip'
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(staging_dir)

        datstr = 'YearPredictionMSD.txt'
        df = pd.read_csv(staging_dir+datstr, header = None)

        y = np.array(df.iloc[:,0]).astype(float)
        X = np.array(df.iloc[:,1:]).astype(float)
    elif ds=='bike':
        uds = fetch_ucirepo(id=275)

        X = np.array(uds['data']['features'].iloc[:,1:])
        y = np.array(uds['data']['targets']).flatten()

    elif ds =='obesity':
        uds = fetch_ucirepo(id=544)

        Xdf = uds['data']['features']
        cat = ['Gender', 'family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
        #Xdf.loc[:,cat]
        #Xdf.drop(cat,axis=1)
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        od = {
                'Insufficient_Weight' : 0,
                'Normal_Weight' : 1,
                'Overweight_Level_I' : 2,
                'Overweight_Level_II' : 3,
                'Obesity_Type_I' : 4,
                'Obesity_Type_II' : 5,
                'Obesity_Type_III' : 6,
                }
        y = np.array(uds['data']['targets'].map(lambda x: od[x])).flatten()
    elif ds=='seoul':
        uds = fetch_ucirepo(id=560)
        dat = uds['data']['features']
        Xdf = dat.iloc[:,2:]
        cat = ['Seasons','Holiday']
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        y = np.sqrt(dat['Rented Bike Count'])
        #y = np.array(pd.get_dummies(uds['data']['targets']).iloc[:,0].astype(float))
    elif ds=='parkinsons':
        uds = fetch_ucirepo(id=189)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']['motor_UPDRS']).flatten()
    elif ds=='aids':
        uds = fetch_ucirepo(id=890)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']).flatten()
    elif ds=='infra':
        uds = fetch_ucirepo(id=925)
        cat = ['Gender','Age','Ethnicity']
        Xdf = uds['data']['features']
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        y = np.array(uds['data']['targets']['aveOralF']).flatten()
    elif ds=='rice':
        uds = fetch_ucirepo(id=545)
        X = np.array(uds['data']['features'])
        y = np.array(pd.get_dummies(uds['data']['targets'],drop_first=True)).flatten().astype(float)
    elif ds=='adult':
        uds = fetch_ucirepo(id=2)
        cat = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
        Xdf = uds['data']['features']
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        y = np.array(pd.get_dummies(uds['data']['targets'],drop_first=True)['income_<=50K.']).flatten().astype(float)
    elif ds=='wine':
        uds = fetch_ucirepo(id=186)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']).flatten().astype(float)
    elif ds=='spam':
        uds = fetch_ucirepo(id=94)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']).flatten().astype(float)
    elif ds=='dropout':
        uds = fetch_ucirepo(id=697)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']=='Dropout').flatten().astype(float)
    elif ds=='shop':
        uds = fetch_ucirepo(id=468)
        cat = ['Month','VisitorType','Weekend']
        Xdf = uds['data']['features']
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        y = np.array(pd.get_dummies(uds['data']['targets'],drop_first=True)).flatten().astype(float)
    else:
        raise Exception("Unknown Dataset!")

    df = pd.DataFrame(X)
    df.columns = ['X'+str(i) for i in range(X.shape[1])]
    df['y'] = y

    if ds=='year': # My rig only has 60 gigs of memory regrettably; if you had 100 you could probably drop this (for running glmnet with 2nd order).
        df = df.sample(frac=1.)
        #df = df.iloc[:df.shape[0]//2,:]
        df = df.iloc[:df.shape[0]//4,:]

    eps = 1e-6
    if ds in class_problems:
        df['y'] = (df['y']-min(df['y'])) / (max(df['y'])-min(df['y']))
    elif ds in reg_problems:
        df['y'] = (df['y'] - np.mean(df['y'])) / (np.std(df['y'])+eps)
    else:
        raise Exception("Unknown dataset type!")
    for i in range(df.shape[1]-1):
        df.iloc[:,i] = (df.iloc[:,i] - np.mean(df.iloc[:,i])) / (np.std(df.iloc[:,i])+eps)

    outname = data_dir+ds+'.csv'
    df.to_csv(outname, index = False)

