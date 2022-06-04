# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:54:47 2021

Thesis Project: Classification and Characterization of the Effect of Migraine 
through Functional Connectivity Characteristics: Application to EEG 
Recordings from a Multimorbid Clinical Sample

-----------------------------------------------------------------------------------------------------------------
@Authors: Ioannis Ziogas & Charalampos Lamprou
Copyright (C) 2022 Ioannis Ziogas and Charalampos Lamprou,SPBTU,ECE,AUTh
-----------------------------------------------------------------------------------------------------------------

Description: In this script a KFold nested Cross-Validation classification scheme is implemented. The input data
should have passed the feature selection stage. The scheme is repeated multiple times, once for each value 
of 'k' best selected features and once for each eye case, for a given FC measure. Evaluation with AUC, Accuracy,
Precision and Recall metrics terminates the procedure. 
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,  cross_val_score, KFold, cross_val_predict,cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,make_scorer,roc_auc_score
import os
import scipy.stats
from itertools import combinations


def castToList(x): #casts x to a list
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]

def mean_confidence_interval(data, confidence=0.95):
    """
    Description: Calculates x% mean confidence interval of the data provided as input.

    Parameters
    ----------
    data : array-like vector or matrix. Columnwise calculation of CI
    confidence : float, 0-1, optional. Level of confidence. The default is 0.95.

    Returns
    -------
    m : float, estimated value (mean).
    m-h: lower CI limit
    m+h: upper CI limit
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a,axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def assemble_dataset(cat,init_path,typList):
    """
    Description: This function gathers all .csv files that belong to the given eye state in the 
    specified path and merges them into a single dataset (EO dataset).
    
    Parameters
    ----------
    cat : str, can be "EO","EC","DiffRatio" or "all". eye state (closed/open/diff-ratio/all)
    init_path : str, path to.csv files (after feature selection).
    typList : list of str, all possible eye states.

    Returns
    -------
    dataset : pandas dataset, Nx(sum_(k)(M_k)+1) rows, where N the number of rows(patients),
    k number of datasets in this state, M_k the number of features of the kth dataset,
    +1 column for the class column(common to all sets)
    """
    data = pd.DataFrame()
    if cat == "EO" or cat == "EC":
        for file in os.listdir(init_path):
            if cat in file: #and feat in file:
                tmpDat = pd.read_csv(os.path.join(init_path,file))
                clas = tmpDat['class']
                tmpDat = tmpDat.iloc[:,:-1]
                cols = tmpDat.columns
                if len(cols) > 0:
                    for typ in typList:
                        if typ in file:
                            newCols = []
                            for c,col in enumerate(cols):
                                newCols.append(col + '_' + typ[0:2])
                            tmpDat.columns = newCols
                    data = pd.concat((data,tmpDat),axis = 1)
                else:
                    os.remove(os.path.join(init_path,file))
    elif cat == "DiffRatio":
        for file in os.listdir(init_path):
            if ('diff' in file or 'ratio' in file):# and feat in file:
                tmpDat = pd.read_csv(os.path.join(init_path,file))
                clas = tmpDat['class']
                tmpDat = tmpDat.iloc[:,:-1]
                cols = tmpDat.columns
                if len(cols) > 0:
                    for typ in typList:
                        if typ in file:
                            newCols = []
                            for c,col in enumerate(cols):
                                newCols.append(col + '_' + typ[0:2])
                            tmpDat.columns = newCols
                    data = pd.concat((data,tmpDat),axis = 1)
                else:
                    os.remove(os.path.join(init_path,file))
    elif cat == "all":
        for file in os.listdir(init_path):
            #if feat in file:
            tmpDat = pd.read_csv(os.path.join(init_path,file))
            clas = tmpDat['class']
            tmpDat = tmpDat.iloc[:,:-1]
            cols = tmpDat.columns
            if len(cols) > 0:
                for typ in typList:
                    if typ in file:
                        newCols = []
                        for c,col in enumerate(cols):
                            newCols.append(col + '_' + typ[0:2])
                        tmpDat.columns = newCols
                data = pd.concat((data,tmpDat),axis = 1)
            else:
                os.remove(os.path.join(init_path,file))
    
    dataset = pd.concat((data,clas),axis = 1)

    return dataset

def initGrids(model_list,random_state):
    """
    Description: This function constructs the grids with the necessary parameters for each model 
    to be used in the parameter optimization process.
    
    Parameters
    ----------
    model_list : list of str, names of classifiers.
    random_state : integer, rng initiator for classifiers where necessary.

    Returns
    -------
    param_grid : list of dictionaries, each dictionary is a grid for a classifier,
    to be searched in the parameter optimization process.
    """
    param_grid = []
    
    if ('SVC' in model_list and 'LinearSVC' in model_list) or ('SVC' in model_list and 'LinearSVC' not in model_list):
        param_grid_svm = {'C': [0.001,0.01,0.1,1,10,100], #10
                'gamma': [0.001,0.01,0.1,1,10,100],
                'kernel': ['rbf'],
                'probability':[True],
                'random_state':[random_state]}
        param_grid.append(param_grid_svm)
        
    if 'LinearSVC' in model_list :        
        param_grid_linear_svm = {'C': [0.001,0.01,0.1,1,10,100],
                'gamma': [0.001,0.01,0.1,1,10,100],
                'kernel': ['linear'],
                'probability':[True],
                'random_state':[random_state]}
        param_grid.append(param_grid_linear_svm)
        
    if 'kNN' in model_list :             
        param_grid_knn = {'n_neighbors':[3,6,9,12,15,18,21],
                            'weights':['uniform','distance'],
                            'metric':['euclidian','manhattan']}
        param_grid.append(param_grid_knn)
        
    if 'LogRe' in model_list:
        param_grid_log = {'C':[0.1, 1, 10, 100, 1000],
                            'penalty':['l1','l2','elasticnet'],
                            'solver':['liblinear'],
                            'random_state':[random_state]}
        param_grid.append(param_grid_log)
        
    if 'DecisionTree' in model_list:
        param_grid_tree = {'criterion':["gini","entropy"],
                                'min_samples_split' : [2,4],
                                'min_samples_leaf': [3,5],
                                'random_state': [random_state],
                                'ccp_alpha': [0,0.1]}
        param_grid.append(param_grid_tree)
        
    if 'GaussianProcess' in model_list:
        param_grid_gpc = {'kernel':[1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()],
                          'max_iter_predict': [100,500],
                          'random_state':[random_state]}
        param_grid.append(param_grid_gpc)
    
    return param_grid

def nestedCV(X,y,results,predictions,selected,scoring,models,param_grid,selec_func,kselec,n_jobs,count,
         random_state = 42,outer_folds = 10,inner_folds = 5):
    """
    Description:
        In this function a nested cross-validation scheme is designed and applied to input 
        data by use of input classifiers. For each classifier, a pipeline with the steps
        of Standard Scaling, feature selection with KBest and training of an optimized- 
        hyperparameters model is constructed. The optimization takes places in the inner loop of 
        the cv process, and the optimized model is incorporated in the pipeline. A standard cross-validation
        scheme is then applied on the data (X,y) with various scorers, and results
        in terms of scores (accuracy,AUC,...), probabilities, and selected features
        is returned, for each classifier used.
        
    Parameters
    ----------
    X : numpy array, NxM, where N number of rows(patients), M number of variables/features.
    
    y : numpy array, Nx1, where N number of rows(patients). class labels for each patient.
    
    results : empty dictionary of dictionaries. Lower level dicts store score values for each model.
    Upper level dict has stores lower level dicts with keys the names of models
    
    predictions : empty dictionary with keys the names of models. Stores probabilities 
    from the predictions made in the cross validation process (in the outer cv loop)
    
    selected : empty list. Stores indices for the best selected features found in each
    split of the outer cv loop.
    
    scoring : dictionary of scoring measures, (measure name - make_scorer function) pair.
    
    models : Classifier Object list, contains classifiers for which the nested_cv 
    process will be executed.
    
    param_grid : list of dictionaries, each dictionary is a grid for a classifier,
    to be searched in the parameter optimization process (grid search, inner cv loop).
    
    selec_func : feature selection criterion, function to be fed as input to
    the SelectKBest selector (sklearn.feature_selection.SelectKBest).
    examples: f_classif (ANOVA F-statistic), mutual_info_classif
    
    kselec : integer, number of features to be selected by SelectKBest.
    
    n_jobs : number of parallel working modules. -1 for all.

    count : necessary counter for initialization of lists in predictions dictionary.
    
    random_state : integer, optional. The default is 42. Random seed initiator,
    used in the random initialization of the random splits in the cv process
    
    outer_folds : integer, optional. The default is 10. Number of outer cv folds
 
    inner_folds : integer, optional. The default is 5. Number of inner cv folds

    Returns
    -------
    results : dictionary of dictionaries. Lower level dicts store score values for each model.
    Upper level dict has stores lower level dicts with keys the names of models.

    predictions : dictionary with keys the names of models. Stores probabilities 
    from the predictions made in the cross validation process (in the outer cv loop).
    
    selected : list. Stores indices for the best selected features found in each
    split of the outer cv loop.

    """
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    for j,model in enumerate(models):

        if model_list[j] == 'NaiveBayes':
            clf = make_pipeline(StandardScaler(), SelectKBest(selec_func, k = kselec), model)
        else:
            grid = GridSearchCV(model, param_grid[j], cv = inner_cv,n_jobs = n_jobs)
            clf = make_pipeline(StandardScaler(), SelectKBest(selec_func, k = kselec), grid)
        scores = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring = scoring)                
        probas = cross_val_predict(clf, X=X, y=y, cv=outer_cv, method = 'predict_proba')[:,1]
            
        if j == 0 and m == 0:
            for train_index, test_index in outer_cv.split(X):
                selector = SelectKBest(selec_func,k = kselec)
                selector.fit(X[train_index,:],y[train_index])
                selected.append(selector.get_support(indices = True))
               
        scores = {key: np.mean(value) for key, value in scores.items()}
        scores.pop('fit_time')
        scores.pop('score_time')
        for key in scores.items():
            if count == 0:
                results[model_list[j]][key[0]] = castToList(scores[key[0]])
                predictions[model_list[j]] = []
                predictions[model_list[j]].append(castToList(probas))
            else:
                results[model_list[j]][key[0]].append(scores[key[0]])
                predictions[model_list[j]].append(castToList(probas))  
    
    return results,predictions,selected

def evaluate(results,predictions,selected,sel_dict,kselec,varnames,model_list):
    """
    Description: In this function the mean 95% Confidence Interval of the scores
    calculated in the nested cv process and stored in the results dictionary, is calculated,
    and stored again in the results dictionary. Also the mean value is taken over the predictions
    made for each patient, resulting in one probability value for each patient, for each model used.
    Finally the indices indicated in the feature selection process, are mapped to the
    names of the features, and stored in the sel_dict dictionary.

    Parameters
    ----------
    results : dictionary of dictionaries. Lower level dicts store score values for each model.
    Upper level dict has stores lower level dicts with keys the names of models.

    predictions : dictionary with keys the names of models. Stores probabilities 
    from the predictions made in the cross validation process
    
    selected : list. Stores indices for the best selected features found in each
    split of the outer cv loop.
    
    sel_dict : dictionary. Each key is a value of best selected features considered
    in the features selection process inside the pipeline of the nested cv. Stores the 
    best selected features indicated for a given k.
    
    kselec : integer, value of k.
    
    varnames : list of str, feature names from initial dataset (class name excluded)

    model_list : list of str, names of classifiers used in the nested cv process.
    Used to search for the results of each model inside the results dictionary.

    Returns
    -------
    results : dictionary of dictionaries. Lower level dicts store upper,lower and mean 95& CI 
    score values for each model. Upper level dict has stores lower level dicts with keys the names of models.
    
    predictions : dictionary with keys the names of models. Stores mean probabilities 
    for each patient, stemming from the predictions made in the cross validation process.
    
    sel_dict : dictionary, stores names of the features, mapped from indices indicated 
    in the feature selection process. Stores the best selected features 
    indicated for a given k.
    """
    
    for model in model_list:
        df = pd.DataFrame()
        for mi in results['LinearSVC'].items():
            CI = []
            meanf,meanf_low,meanf_high = mean_confidence_interval(
                results[model][mi[0]], confidence=0.95)
            if meanf_high > 1:
                meanf_high = 1
            CI.append(meanf_low)
            CI.append(meanf)
            CI.append(meanf_high)
            df[mi[0]] = CI
        results[model] = df
        
        predictions[model] = np.mean(predictions[model], axis = 0)
    
    "Save Selected"
    values, counts = np.unique(selected, return_counts=True) 
    if not len(counts) == kselec:
        ind = np.argpartition(-counts, kth=kselec)[:kselec]	
    else:
        ind = np.argpartition(-counts, kth=kselec-1)[:kselec]	
    varinds = values[ind]
    selvars = [varnames[varind] for varind in varinds]
    sel_dict['k'+str(kselec)] = {'selected': selected,
                    'varNames':varnames,
                    'varIndices':varinds,
                    'values': values,
                    'counts': counts}
    
    return results,predictions,sel_dict


"Initializations"

pwd = os.getcwd()
category  = 'crossBspec_anti' #'COH','PLV','crossBspec','PAC','PAC_tf'
init_path = os.path.join('feature_selection_results',category)
dest_folder = 'KFoldResults_MI' #_MI
dest_folder = os.path.join(pwd,dest_folder)
sets_folder = 'newDatasets_MI' #_MI
sets_folder = os.path.join(pwd,sets_folder)
inner_dest_folder = category
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)
dest_folder = os.path.join(dest_folder,inner_dest_folder)
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)
if not os.path.exists(sets_folder):
    os.mkdir(sets_folder)
sets_folder = os.path.join(sets_folder,inner_dest_folder)
if not os.path.exists(sets_folder):
    os.mkdir(sets_folder)
save_selected = True
save_results = True

rand1 = 1
rand2 = 50#50
folds = 10
#kstart = 'all'#10
#kstep = 5
n_jobs = -1

scoring = {'accuracy' : make_scorer(accuracy_score),
           'Recall' : make_scorer(recall_score),
           'Precision': make_scorer(precision_score),
           'AUC' : make_scorer(roc_auc_score)
              }

model_list = ["SVC","LinearSVC","kNN","LogRe","DecisionTree","GaussianProcess"]#,"NaiveBayes"]
models = [SVC(),SVC(), KNeighborsClassifier(), LogisticRegression(),DecisionTreeClassifier(),
          GaussianProcessClassifier()]#,GaussianNB()]

kstep = 5
selec_func = mutual_info_classif #mutual_info_classif,f_classif
#feat = 'crossBspec'
typList = ["EO","EC","ratio","difference"]
categ = ['EO','EC'] #'DiffRatio',
kstartvec = [5,5,5]
expand = 'up' #'up','down'
maxkvec = [75,75,75]

for k,cat in enumerate(categ):
    
    maxk = maxkvec[k]
    kstart = kstartvec[k]
    #caseList = ["inter","intra","left_right","anti-symmetric"]
    fname = category + '_' + cat + '_' + 'Dataset' #feat
    
    "If results are going to be calculated for the first time"
    if not os.path.exists(os.path.join(dest_folder,fname + '.npy')):
        
        dataset = assemble_dataset(cat,init_path,typList)
          
        dataset.to_csv(os.path.join(sets_folder,fname + '.csv'),index = False)
        
        varnames = list(dataset.iloc[:,:-1].columns)
        
        X = dataset.iloc[:,0:-1].values
        y = dataset.iloc[:,-1].values
        
        featureNumber = len(X[1,:])
        if featureNumber > maxk:
            featureNumber = maxk
        
        #kstart = featureNumber
        
        if kstart == 'all':
            kvec = [len(X[1,:])]
        else:
            if kstart < featureNumber:
                kvec = np.arange(start = kstart,stop = featureNumber,step = kstep)
            else:
                kvec = [featureNumber]
            if kvec[-1] != featureNumber:
                kvec = np.append(kvec,featureNumber)
        
        allResults = {}
        
        sel_dict = {'k'+str(k): {} for k in kvec}
        
        for kselec in kvec:
            
            results = {}
            
            selected = []
            
            for model in model_list:
                results[model] = {}
            
            predictions = {}
            for model in model_list:
                predictions[model] = {}
            
            #nested_scores = np.zeros((rand1*rand2,len(models)))
            count = 0
            for m in range(rand1):
                
                param_grid = initGrids(model_list,random_state = m)
                           
                for n in range(rand2):
                    results,predictions,selected = nestedCV(X,y,results,predictions,selected,scoring,
                        models,param_grid,selec_func,kselec,n_jobs,count, 
                            random_state = n,outer_folds = folds,inner_folds = 5)
                      
                    count = count + 1
            
            results,predictions,sel_dict = evaluate(results,
                predictions,selected,sel_dict,kselec,varnames,model_list)
            
            totalResults = {}
            totalResults['k'] = kselec
            totalResults['results'] = results
            totalResults['probabilities'] = predictions
            totalResults['selected'] = selected
            allResults['k'+str(kselec)] = totalResults
        
        "Save and exit"
        if save_results:    
            np.save(os.path.join(dest_folder,fname + '.npy'),allResults)
        if save_selected:
           if not os.path.exists(dest_folder):
               os.mkdir(dest_folder)
           sel_file = os.path.join(dest_folder,fname + ' selected.npy')
           np.save(sel_file,sel_dict)    

    "/////////////////////////////////////////////////////////////////////////"
    "/////////////////////////////////////////////////////////////////////////"
    "/////////////////////////////////////////////////////////////////////////"

    """In this section, the process is implemented in case there already exists
    a file with results from a previous run. In this case, if the user specifies
    different k values to calculate results for, the new results are incorporated into
    the already existing file.
    """

    "If results have been already calculated for this dataset"
    if os.path.exists(os.path.join(dest_folder,fname + '.npy')):
        
        existingRes = np.load(os.path.join(dest_folder,fname + '.npy'), allow_pickle = True).item()
        allkeys = [*existingRes]
        highk = existingRes[allkeys[-1]]['k']
        lowk = existingRes[allkeys[0]]['k']
        
        dataset = pd.read_csv(os.path.join(sets_folder,fname + '.csv'))
        
        X = dataset.iloc[:,0:-1].values
        y = dataset.iloc[:,-1].values
        
        featureNumber = len(X[1,:])
        
        if expand == 'down':
            kstart = kstartvec[k]
            featureNumber = lowk - kstep
        elif expand == 'up':
            if featureNumber <= highk:
                continue
            kstart = highk + kstep #start from n+1, where n: last k of previous run
            if featureNumber > maxk:
                featureNumber = maxk
        
        varnames = list(dataset.iloc[:,:-1].columns)
    
        if kstart == 'all':
            kvec = [len(X[1,:])]
        else:
            if kstart < featureNumber:
                kvec = np.arange(start = kstart,stop = featureNumber,step = kstep)
            else:
                kvec = [featureNumber]
            if kvec[-1] != featureNumber:
                kvec = np.append(kvec,featureNumber)
        
        allResults = {}
        new_sel_dict = {'k'+str(k): {} for k in kvec}
        
        
        for kselec in kvec:
            
            results = {}
            
            selected = []
            
            for model in model_list:
                results[model] = {}
            
            predictions = {}
            for model in model_list:
                predictions[model] = {}
            
            #nested_scores = np.zeros((rand1*rand2,len(models)))
            count = 0
            for m in range(rand1):
                
                param_grid = initGrids(model_list,random_state = m)
                
                for n in range(rand2):
                    results,predictions,selected = nestedCV(X,y,results,predictions,selected,
                        scoring, models,param_grid,selec_func,kselec,n_jobs,count,
                            random_state = n,outer_folds = folds,inner_folds = 5)
                    
                    count = count + 1
            
            results,predictions,new_sel_dict = evaluate(results,
                predictions,selected,new_sel_dict,kselec,varnames,model_list)
        
            totalResults = {}
            totalResults['k'] = kselec
            totalResults['results'] = results
            totalResults['probabilities'] = predictions
            totalResults['selected'] = selected
            allResults['k'+str(kselec)] = totalResults
        
        existingRes.update(allResults)
        sel_dict = np.load(os.path.join(dest_folder,fname + ' selected.npy'),allow_pickle = True).item()
        sel_dict.update(new_sel_dict)
        
        "Save and exit"
        if save_results:    
            os.remove(os.path.join(dest_folder,fname + '.npy'))
            np.save(os.path.join(dest_folder,fname + '.npy'),existingRes)
        if save_selected:
           if not os.path.exists(dest_folder):
               os.mkdir(dest_folder)
           sel_file = os.path.join(dest_folder,fname + ' selected.npy')
           os.remove(sel_file)
           np.save(sel_file,sel_dict)           
    
    
