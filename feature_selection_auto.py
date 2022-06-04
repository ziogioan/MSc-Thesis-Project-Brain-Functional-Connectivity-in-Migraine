# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:58:53 2021

Thesis Project: Classification and Characterization of the Effect of Migraine 
through Functional Connectivity Characteristics: Application to EEG 
Recordings from a Multimorbid Clinical Sample

-----------------------------------------------------------------------------------------------------------------
@Authors: Ioannis Ziogas & Charalampos Lamprou
Copyright (C) 2022 Ioannis Ziogas and Charalampos Lamprou,SPBTU,ECE,AUTh
-----------------------------------------------------------------------------------------------------------------

Description: In this script, a pipeline designed to isolate the best performing variables/features
is implemented. 
1. As a first step, missing values (NaN's) are taken care of with imputation techniques. 
2. Then a "statistical filter" is applied: 
    2.1. each feature is tested for its statistical significance, using
        as grouping variable the presence of the desired pathology (e.g. Migraine). 
    2.2. each feature is tested for its statistical significance when considering 
        as (a) grouping variable(s) a pathology label vector. This is repeated for one, 
        two and three pathologies at the same time. Features that do meet the set statistical 
        significance when tested with a pathology or a combination of pathologies, 
        other than the desired pathology (e.g. Migraine), are discarded, to ensure that 
        the effect of the desired pathology is faithfully represented in the remaining features.
    2.3. each feature is tested for its statistical significance, using as grouping variable the age of 
        each subject, whether they are under or over a certain age. This is done to ensure that age
        is not a factor that statistically affects a feature. Features that are statistically dependent
        on age, are discarded. 

3. Finally a feature selection stage with voting is implemented. The remaining from the previous steps
features are fed to different feature selection algorithms. Each algorithm "selects" a number of features
as the best. In the end of the procedure, each algorithm "votes" for the best features, and the 
final set of "best features" is made up from the features that gathered the most "votes".
"""

import pandas as pd
import numpy as np
import numpy.matlib
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.linear_model import LassoCV,RidgeClassifier,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,ExtraTreesRegressor
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from collections import OrderedDict
from functools import partial
import os
import scipy.stats
from itertools import combinations
import warnings


def castToList(x): #casts x to a list
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]
    
def class_histogram(X,y,class_names,kind):
    """
    Description: Separates rows of data matrix X according to class label. 
    Then for each variable/feature, plot a histogram to compare distribution
    of each variable into each class.
    
    Parameters
    ----------
    X : data matrix, N subjects x M features.
    y : class labels vector.
    class_names : names of classes (migraine - control).

    Returns
    -------
    None.

    """
    data = pd.concat((X,y),axis = 1)
    n0 = class_names[0]
    n1 = class_names[1]
    rows1 = np.where(data['class'] == 1)
    rows0 = np.where(data['class'] == 0)
    data['class_name'] = data['class']
    data['class_name'].iloc[rows1] = n1
    data['class_name'].iloc[rows0] = n0
    
    cols = data.shape[1]
        
    for col in data.columns[:-2]:
        hs = pd.concat((data['class_name'],data[col]),axis = 1)
        fig = plt.figure()
        if kind == 'boxplot':
            sns.boxplot(x = "class_name", y = col, 
                        data=hs, hue = "class_name")
            plt.title(name + ' Boxplot: ' + col)
        elif kind == 'histogram':
            sns.histplot(data = hs,element = "step",x = col,
                     hue = 'class_name',bins = int(data.shape[0]/2 - 15))
            plt.title(name + ' Histogram: ' + col)
        plt.show()

    return None

def concat_data(dataset_class1,dataset_class0):
    """
    Description: This function receives as input two array-likes, having the same variables and
    belonging to different classes, and merges them in a single array.
    
    Parameters
    ----------
    dataset_class1 : array-like, data of class 1.
    dataset_class0 : array-like, data of class 0.

    Returns
    -------
    data : array-like, concatenated dataset.
    
    """
    rows1 = dataset_class1.shape[0]
    rows0 = dataset_class0.shape[0]

    data = pd.concat((dataset_class1,dataset_class0),axis = 0,ignore_index = True)
    rows = data.shape[0]
    cols = data.shape[1]
    
    class1 = np.matlib.repmat(1,rows1,1).reshape((rows1,))
    class0 = np.matlib.repmat(0,rows0,1).reshape((rows0,))
    
    clas = np.concatenate((class1,class0),axis=0)
    clas = pd.DataFrame(clas)
    clas = clas.rename(columns = {0:"class"})
    
    data = pd.concat((data,clas),axis = 1)
    
    return data

def clean_data(data,n_jobs,NaNthresh = 0.2,fillna_strategy = 'mean'):
    """
    Description: This function cleans data from NaN values. First drops NaN values columnwise
    according to given threshold (maximum percentage of NaN values),then fills
    remaining NaN values according to fillna_strategy. Input data must not contain the class column
    
    Parameters
    ----------
    data : array-like, N subjects x M variables.

    n_jobs : int, number of parallel cores for the IterativeImputer algorithm. -1 is for all cores.
    
    NaNthresh : float, optional. Maximum percentage (0-1) of NaN values that a column can
    have without being discarded. Columns that exceed that percentage are discarded. The default is 0.2.
    
    fillna_strategy : str, optional. Strategy to fill the missing values with the imputation algorithms
    (Simple or Iterative). The default is 'mean'. 
    Options: {‘mean’, ‘median’, ‘most_frequent’, ‘constant’}

    Returns
    -------
    datacl : data after cleaning.
    colna : array-like. Vector containing counts of NaN values in columns that had NaNs.
    imputation : str. Indicates type of imputation used ('iterative' or 'simple').
    """
    
    cols = data.shape[1]
    rows = data.shape[0]
    drop_int = round(NaNthresh*rows)
    
    na = data.isna() #Boolean mask of NaNs
    
    #Count NaNs and not-NaNs in each column
    colna = pd.DataFrame()
    for i in range(cols):
        colna = pd.concat((colna,na.iloc[:,i].value_counts()),axis = 0)
        
    datadr = data.dropna(axis = 1,how = 'any',thresh = rows-drop_int,inplace = False)
    coln2 = datadr.columns
    datadr = datadr.values
    
    t = colna.index

    colna = colna[t] #Only trues
    if len(colna) != 0:
        mostNaNs = max(colna[0])
    else:
        mostNaNs = 0
        
    if mostNaNs < 5:
        imputation = "simple"
    else:
        imputation = "iterative"
        
    if imputation == "simple":
        imputer = SimpleImputer(missing_values = np.nan, strategy = fillna_strategy)
    elif imputation == "iterative":
        imputer = IterativeImputer(estimator = ExtraTreesRegressor(n_estimators = 10,random_state = 0,n_jobs = n_jobs),
                random_state = 0,max_iter = 10,tol = 0.001,initial_strategy= fillna_strategy,
                missing_values = np.nan)
        
    imputer = imputer.fit(datadr)
  
    datacl = imputer.transform(datadr)
    datacl = pd.DataFrame(datacl)
    datacl.columns = coln2
    
    return datacl,colna,imputation

def keep_common_cols(data1,data0):
    """
    Description: This function receives as input two array-likes, and keeps for each dataset 
    only the columns that have common variable names.
    
    Parameters
    ----------
    data1 : array-like, input dataset 1.
    data0 : array-like, input dataset 0.

    Returns
    -------
    data1 : array-like, output dataset 1.
    data0 : array-like, output dataset 0..

    """
    cols1 = data1.columns
    cols0 = data0.columns
    common_cols = set(cols1).intersection(set(cols0))

    new_cols1 = [col for col in cols1 if col in common_cols]
    new_cols0 = [col for col in cols0 if col in common_cols]

    data1 = data1[new_cols1]
    data0 = data0[new_cols0]

    return data1,data0

def feature_selection(X,y,feature_names,**kwargs):
    """
    Description: In this function, different feature selection algorithms 
    are available for implementation. 

    Parameters
    ----------
    X : array-like, input dataset.
    y : array-like vector, class labels for each subject.
    feature_names : pandas dataframe, list of column (feature) names.
    **kwargs : additional arguments, e.g. 'alg' - algorithm name.  

    Raises
    ------
    Exception
        if specified algorithm name is not one of those included in this function.

    Returns
    -------
    Output arguments, depend on which algorithm is used.
    """
    
    alg = kwargs.get('alg')
    if alg == 'VarianceThreshold':
        "VarianceThreshold"
        "Inputs: threshold - variance of a feature must exceed this threshold,"
        "or else it is removed"
        "Outputs:transformed Xnew"
        
        from sklearn.feature_selection import VarianceThreshold
        
        threshold = kwargs.get('threshold')
        selector = VarianceThreshold(threshold = threshold)
        Xnew = selector.fit_transform(X)
        variances = selector.variances_
        
        return Xnew,variances
    
    elif alg == 'KBest':
        "Select K Best"
        "Inputs: kscore_func,ksel"
        "Outputs: Scores of best features, names of best features,transformed Xnew "
        
        from sklearn.feature_selection import SelectKBest
        
        kscore_func = kwargs.get('kscore_func')
        ksel = kwargs.get('ksel')
        KBest = SelectKBest(kscore_func, k = ksel)
        Xnew = KBest.fit_transform(X, y)
        support = KBest.get_support()
        scores = KBest.scores_
        inds = np.argpartition(scores,-ksel)[-ksel:] #indexes of best features
        kbest_scores = np.sort(scores)[-ksel:] #best features
        kbest_names = feature_names[support]
        
        return kbest_scores,kbest_names,Xnew
    
    elif alg == 'LassoCV':
    
        "LASSO Cross-Validation"
        "Inputs: n_features,tol,max_iter,cv,random_state"
        "Outputs: scores,names"
    
        from sklearn.linear_model import LassoCV    
    
        n_features = kwargs.get('n_features'); tol = kwargs.get('tol') 
        max_iter = kwargs.get('max_iter'); random_state = kwargs.get('random_state')
        cv = kwargs.get('cv')
        lasso = LassoCV(tol = tol,max_iter = max_iter,cv = cv,
            random_state = random_state).fit(X, y)
        importance = np.abs(lasso.coef_)
        plt.bar(height = importance, x = feature_names)
        plt.title("LASSO - Feature importances via coefficients")# + name)
        plt.xticks(rotation=45)
        plt.tick_params(labelsize = 7)
        plt.show()
        
        inds = np.argpartition(importance,-n_features)[-n_features:] #indexes of best features
        important_scores = np.sort(importance)[-n_features:] #best features
        important_names = feature_names[inds]
        
        "Transform method can be used HERE if needed"
        
        return important_scores,important_names
    
    elif alg == 'RFE':
        
        "Recursive Feature Elimination"
        "Inputs: estimator,n_features"
        "Outputs: rfeNames,Ranking"
        
        from sklearn.feature_selection import RFE
        
        estimator = kwargs.get('estimator') 
        n_features= kwargs.get('n_features')
        selector = RFE(estimator = estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X,y)
        support = selector.support_
        ranking = selector.ranking_
        
        rfeNames = feature_names[support]
        
        return rfeNames,ranking
    
    elif alg == 'RFECV':
        
        "Recursive Feature Elimination with Cross-Validation"
        "Inputs: estimator,min_features_to_select,cv,scoring"
        "Outputs: plot,opt_n_features,rfecvNames"
        
        from sklearn.feature_selection import RFECV
        
        estimator = kwargs.get('estimator')
        min_features_to_select = kwargs.get('min_features_to_select')
        if not kwargs.get('cv'):
            cv = 5
        else:
            cv = kwargs.get('cv')
        if not kwargs.get('scoring'):
            scoring = 'accuracy'
        else:
            scoring = kwargs.get('scoring')
            
        rfecv = RFECV(estimator = estimator,step = 1,cv = cv,scoring = scoring,
                min_features_to_select = min_features_to_select).fit(X,y)
        cv_support = rfecv.support_
        cv_ranking = rfecv.ranking_
        grid_scores = rfecv.grid_scores_
        if len(np.shape(grid_scores)) > 1:
            grid_scores = np.mean(grid_scores,axis = 1)
        
        rfecvNames = feature_names[cv_support]
        
        opt_n_features = rfecv.n_features_
        
        "Plot number of features VS. cross-validation scores"
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (accuracy)")
        plt.plot(range(min_features_to_select,
                       len(grid_scores) + min_features_to_select),
                 grid_scores)
        plt.show()
        
        return opt_n_features,rfecvNames,grid_scores
    
    elif alg == 'ExtraTrees':
        
        "ExtraTreesClassifier"
        "Inputs: n_estimators,criterion,max_features,random_state,n_features"
        "Outputs: importance,names"
        
        from sklearn.ensemble import ExtraTreesClassifier
        
        n_features = kwargs.get('n_features')
        n_estimators = kwargs.get('n_estimators')
        if not kwargs.get('criterion'):
            criterion = "gini"
        else:
            criterion = kwargs.get('criterion')
        random_state = kwargs.get('random_state')
        max_features = kwargs.get('max_features')
        
        extraTrees = ExtraTreesClassifier(n_estimators = n_estimators,
                    criterion = criterion,max_features = max_features,
                    n_jobs = -1,random_state = random_state,verbose = 0)
        
        extraTrees = extraTrees.fit(X,y)
        trees_imp = extraTrees.feature_importances_
        
        inds = np.argpartition(trees_imp,-n_features)[-n_features:] #indexes of best features
        trees_imp = np.sort(trees_imp)[-n_features:] #best features
        trees_names = feature_names[inds]
        
        return trees_imp,trees_names
        
    elif alg == 'SelectFromModel':
        "Estimator + Select From Model"
        "Inputs: prefit estimator if prefit = True"
        "Outputs: clf_features - features selected based on classifier , Xnew"
        
        from sklearn.feature_selection import SelectFromModel
        
        estimator = kwargs.get('estimator')
        model = SelectFromModel(estimator,prefit = True)
        clf_features = model.get_support()
        clf_features = feature_names[clf_features]
        Xnew = model.transform(X)
        
        return clf_features,Xnew
           
    elif alg == 'PermutationImportance':
        
        "Permutation Importance"
        "Inputs: fitted classifier, n_repeats,random_state, permMeanThreshold"
        "Outputs: perm_imp_features,perm_imp_mean(for each feature) perm_imp_std"
        
        from sklearn.inspection import permutation_importance
        
        classifier = kwargs.get('classifier')
        n_repeats = kwargs.get('n_repeats')
        random_state = kwargs.get('random_state')
        permMeanThreshold = kwargs.get('permMeanThreshold')
        
        permObj = permutation_importance(classifier,X,y,n_repeats = 100,random_state = 0)
        perm_imp_mean = permObj.importances_mean
        perm_imp_std = permObj.importances_std

        perm_imp_df = pd.Series(perm_imp_mean,index = feature_names)

        fig, ax = plt.subplots()
        perm_imp_df.plot.bar(yerr = perm_imp_std, ax = ax)
        ax.set_title("Feature importances using PermutationImportance")
        ax.set_ylabel("Mean decrease in baseline metric")
        fig.tight_layout()
        plt.show()

        perm_imp_features = []
        perm_imp_inds = []
        inds = perm_imp_mean.argsort()[::-1]
        for i in inds:
            if perm_imp_mean[i] - 2*perm_imp_std[i] > 0 and perm_imp_mean[i] >= permMeanThreshold:
                perm_imp_inds.append(i)
        
        perm_imp_features = feature_names[perm_imp_inds]
        perm_imp_mean = perm_imp_mean[perm_imp_inds]
        perm_imp_std = perm_imp_std[perm_imp_inds]
        
        return perm_imp_features,perm_imp_mean,perm_imp_std
        
    elif alg == 'SequentialFeatureSelector':    
        
        "SequentialFeatureSelector"
        "Inputs: estimator,n_features_to_select,direction"
        "Outputs: names"
        
        from sklearn.feature_selection import SequentialFeatureSelector
        
        estimator = kwargs.get('estimator')
        n_features_to_select = kwargs.get('n_features_to_select')
        direction = kwargs.get('direction')
        sfs = SequentialFeatureSelector(estimator = estimator,
            n_features_to_select = n_features_to_select,direction=direction).fit(X,y)
                
        names = feature_names[sfs.get_support()]
        
        return names
    
    else:
        
        raise Exception("Specified algorithm does not exist in this function")

        return None

def pvalFilter(data,class_name,statTests = ["t-test","Wilcoxon"],
                   signifThresh = 0.05,pvalSuccessRatio = 2/2,
                       ImbRatio = 1/4,min_obs = 20):
    """
    Description: This function filters features based on their p-values, checking for statistical significance 
    using the variable class_name.
    If a feature can pass the significance threshold in a number of statistical 
    tests (statTests var, array of strings), which is defined by the pvalSuccessRatio
    variable, then this feature is considered significant. For imbalanced classes that 
    exceed a certain imbalance ratio, pvalFilter is not executed 

    Parameters
    ----------
    data : pandas dataframe, input data.
    
    class_name : str, name of class column, as stated in the 'data' dataframe.
    
    statTests : list of str, optional. Names of statistical tests to be implemented.
        The default is ["t-test","Wilcoxon"].
        
    signifThresh : float, optional. Statistical significance p-value threshold. The default is 0.05.
    
    pvalSuccessRatio : ratio, optional. Wanted success ratio of statistical tests. The default is 2/2.
    
    ImbRatio : ratio, optional. Minimum acceptable imbalance ratio between the two classes. The default is 1/4.
    
    min_obs : int, optional. Minimum class observations for statistical tests to be applied. The default is 20.

    Returns
    -------
    datafin : pandas dataframe, output dataset, containing variables that meet the set statistical criteria.
    
    colnames : str, names of variables that met the set statistical criteria.
    
    pvalMask : boolean mask of all variables according to whether they met the set statistical criteria.
    """
    
    colnames = data.columns
    rowsClass0 = np.where(data[class_name] == 0)[0]
    rowsClass1= np.where(data[class_name] == 1)[0]
    c0obs = len(rowsClass0)
    c1obs = len(rowsClass1)
    
    pvalMask= []
    if c0obs == 0 or c1obs == 0 or (c0obs + c1obs < min_obs)  :
        warnings.warn('A class has 0 positive or negative observations for  '
                      + class_name +'. pvalFilter will not be executed for this class'
                      ,UserWarning)
        colnames = pd.Index([])
        return data,colnames, pvalMask
    else:
        if c0obs/c1obs < ImbRatio or c1obs/c0obs < ImbRatio:
            warnings.warn('Minimum acceptable class proportion was not achieved for class '
                          + class_name +'. pvalFilter will not be executed for this class'
                          ,UserWarning)
            colnames = pd.Index([])
            return data, colnames, pvalMask
        else:
            
            for clmn in colnames[:-1]:
                pcounter = 0
                fData = data[clmn]
                featClass0 = fData.iloc[rowsClass0]   
                featClass1 = fData.iloc[rowsClass1]  
                if not (np.var(featClass0) == 0 or np.var(featClass1) == 0):
                    
                    "t-test"
                    if "t-test" in statTests:
                        [tStat,pvalt] = scipy.stats.ttest_ind(featClass1, featClass0, equal_var=True)
                        if pvalt <= signifThresh:
                            pcounter = pcounter + 1
                    "Welch's t-test"
                    if "Welch's t-test" in statTests:
                        [tStatW,pvalW] = scipy.stats.ttest_ind(featClass1, featClass0, equal_var=False)
                        if pvalW <= signifThresh:
                            pcounter = pcounter + 1
                    "Mann-Whitney U Test / Wilcoxon Rank-Sum Test"
                    if "Mann-Whitney" in statTests:
                        try:
                            [tStatWx,pvalWx] = scipy.stats.mannwhitneyu(featClass1, featClass0)
                        except:
                            a = 1
                        if pvalWx <= signifThresh:
                            pcounter = pcounter + 1
                    "Wilcoxon signed-rank test"
                    if "Wilcoxon" in statTests:
                        [tStatMWU,pvalMWU] = scipy.stats.wilcoxon(x = featClass1, y = featClass0, zero_method='wilcox')
                        if pvalMWU <= signifThresh:
                            pcounter = pcounter + 1                
              
                else:
                    pcounter = len(statTests) #If a feature has 0 variance, it must be removed
                        
                if pcounter/len(statTests) >= pvalSuccessRatio:
                    pvalMask.append(True)
                else:
                    pvalMask.append(False)
            
            pvalMask.append(True) #For the class column
            colnames = colnames[pvalMask]
            datafin = data[colnames]
            
            return datafin, colnames, pvalMask

"INITIALIZATIONS"
"----------------------------------------------------------------------------------------------------------------------"
"Algorithms' parameters"
scoreRfecvThreshold = 0.65 #Threshold for RFECV: min grid score required
featRfecvThreshold = 50 #Threshold for RFECV: max number of features allowed 
permMeanThreshold = 0.1 #Threshold for PermutationImportance: min{mean decrease} in baseline metric allowed
NaNthresh = 0.1 #Threshold for maximum allowed NaN values (percentage)
n_features = 10 #Number of features considered by each algorithm
rand = 10 #Random states, where needed
min_max_scaling = True #Min-Max Scaling before VarianceThreshold
var_thresh = 0.01 # std = 0.12 e.g0.0024 -> columns with std < 0.05 will be removed
save_new_dataset = True
n_jobs = -1

"Statistical tests parameters"
statTests = ["Welch's t-test","Mann-Whitney"]#,"Wilcoxon","Welch's t-test","Mann-Whitney"]
signifThresh = 0.01 # 99% significance
pvalSuccessRatio = 2/2 #2/3
pvalSuccessRatioStep1 = 2/2
pvalSuccessRatioStep2 = 2/2
comorbidityThresh = 0 #Maximum accepted comorbidities other than migraine
#correlated via p-value with a feature
min_obs = 20 #minimum number of observations(rows) in a set to apply a statistical test
ImbRatio = 1/4 #Minimum acceptable imbalance between classes at the time of testing

"Selection of EEG parameters"
typlist = ['EC','EO','ratio','difference']#'EO','EC',...
caselist = ['inter','intra','left_right','anti-symmetric']# EEG regional scenaria
featlist = ['COH','PLV','crossBspec','PAC']# available connectivity measures list
conds = ['cont','mig']
category = 'crossBspec_anti' #Connectivity measure of interest - [COH,PLV,crossBspec_anti,PAC]
feat = category
AgeLim = 'Under12' #Age limit to check for statistical differences between age groups
pathology = 'MIG' #Pathology to check for age differences

"Paths"
path_init = os.getcwd()#current script path
dest_folder = os.path.join(path_init,'feature_selection_results')
clean_folder = os.path.join(path_init,'cleanedDatasets')
dest_folder = os.path.join(dest_folder,category)

if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

read_folder = os.path.join(path_init,feat)

"Load Classes csv"

classes = pd.read_csv('Classes.csv')
del classes['STR'] #Empty class

comorbid = classes.columns
"----------------------------------------------------------------------------------------------------------------------"
"Load matching Migraine and Control datasets sequentially for EC/EO/..., inter/intra/...,COH/PLV/... "
#Iterate through dataset files inside the specified category folder
for file in os.listdir(read_folder):
    readcount = 0 #Counter to determine if the correct datasets have been collected
    # If it is equal to 2, then everything is fine 
    fullfile = os.path.join(read_folder,file) #Fullpath of the loaded file
    if not os.path.isdir(fullfile): #Check that file is not a folder
        #Gather the information of the current file
        if category in file:
            for typ in typlist:
                if typ in file:
                    temptyp = typ
            for case in caselist:
                if case in file:
                    tempcase = case
            for feat in featlist:
                if feat in file:
                    tempfeat = feat
            for cond in conds:
                if cond in file:
                    tempcond = cond
            
            if tempcond == conds[1]: #If contains mig
                data1 = pd.read_csv(fullfile) #class1: migraine
                readcount = readcount + 1
                readname1 = file[:-4]    #Keep filename except '.csv'             
                
                #Find the cond file
                for file2 in os.listdir(read_folder):
                    fullfile2 = os.path.join(read_folder,file2)
                    if not os.path.isdir(fullfile2): #Check that file is not a folder
                        #Check that file information matches exactly the first file information
                        if (temptyp in file2) and (tempcase in file2) and (tempfeat in file2) and (conds[0] in file2):
                                data0 = pd.read_csv(fullfile2) #class0: control
                                readcount = readcount + 1
                                readname0 = file2[:-4]
                                name = category + '_' + tempcase + '_' + temptyp
    "----------------------------------------------------------------------------------------------------------------------"                                
    if readcount > 2:
        raise Exception("Something went wrong")
    elif readcount <2: #If matching mig and cont files were not found in this iteration
        pass
    elif readcount == 2: #If matching mig and cont files were found in this iteration
        "Remove div (EC/EO) features from anti-symmetric cases"
        featureNames1 = data1.columns[:-1]
        fnMask1 = []
        for fn1 in featureNames1:
            if 'div' in fn1:
                fnMask1.append(False)
            else:
                fnMask1.append(True)
        featureNames1 = featureNames1[fnMask1]
        data1 = data1[featureNames1]
        
        featureNames0 = data0.columns[:-1]
        fnMask0 = []
        for fn0 in featureNames0:
            if 'div' in fn0:
                fnMask0.append(False)
            else:
                fnMask0.append(True)
        featureNames0 = featureNames0[fnMask0]
        data0 = data0[featureNames0]
        
        desc1 = data1.describe(include = "all")
        desc0 = data0.describe(include = "all")
    
        "/////////////////////////////CLEANING////////////////////////////////////"
        "NaN removal and imputation"
        dictclean1 = {'data': [],'imp': []}; dictclean0 = {'data': [],'imp': []};
        cleanname1 = readname1 + '_' + 'cleaned'
        cleanname0 = readname0 + '_' + 'cleaned'
        if not os.path.exists(os.path.join(clean_folder,cleanname1 + '.npy')):
            data1cl,colna1,imputation1 = clean_data(data1,n_jobs,NaNthresh = NaNthresh)
            dictclean1['data'] = data1cl
            dictclean1['imp'] = imputation1
            np.save(os.path.join(clean_folder,cleanname1 + '.npy'),dictclean1)
        else:
            dictclean1 = np.load(os.path.join(clean_folder,cleanname1+ '.npy'),
                                 allow_pickle='TRUE').item() 
            data1cl = dictclean1['data']  
            imputation1 = dictclean1['imp']  
            
        if not os.path.exists(os.path.join(clean_folder,cleanname0 + '.npy')):
            data0cl,colna0,imputation0 = clean_data(data0,n_jobs,NaNthresh = NaNthresh)
            dictclean0['data'] = data0cl
            dictclean0['imp'] = imputation0
            np.save(os.path.join(clean_folder,cleanname0 + '.npy'),dictclean0)
        else:
            dictclean0 = np.load(os.path.join(clean_folder,cleanname0+ '.npy'),
                                 allow_pickle='TRUE').item() 
            data0cl = dictclean0['data']  
            imputation0 = dictclean0['imp']  
            
        
        if imputation1 != imputation0:
            imputation = "mixed"
        else:
            imputation = imputation1
        
        desc1cl = data1cl.describe(include = "all")
        desc0cl = data0cl.describe(include = "all")
        
        "Keep common columns after cleaning"
        data1c,data0c = keep_common_cols(data1cl,data0cl)
        
        desc1cc = data1c.describe(include = "all")
        desc0cc = data0c.describe(include = "all")
        
        "Concatenate"
        data = concat_data(data1c,data0c)
        
        feature_names = data.iloc[:,:-1].columns
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        
        desctot = data.describe(include = "all")
        
        "/////////////////////////////VARIANCE THRESHOLD//////////////////////////"
        if min_max_scaling:
            "Min-Max Scaling"
            mmScaler = MinMaxScaler()
            X = mmScaler.fit_transform(X)
        
        datammSc = pd.DataFrame(data = X,columns = feature_names)
        descmmSc = datammSc.describe(include ="all")
        "Apply Variance Threshold to Scaled Data and Inverse Transform"
        _,variances = feature_selection(X,y,feature_names,
                    alg = 'VarianceThreshold',threshold = var_thresh)
        inds_out = np.where(variances < var_thresh)
        left_out_features = feature_names[inds_out]
        inds_in = np.array(np.where(variances > var_thresh))
        inds_in = inds_in.reshape((len(inds_in[0]),))
        feature_names = feature_names[inds_in]
        if min_max_scaling:
            "Inverse Min-Max Scaling"
            X = mmScaler.inverse_transform(X)
        
        try:
            Xvar = np.squeeze(X[:,inds_in]) 
        except:
            Xvar = np.squeeze(X.iloc[:,inds_in])
        datavar = pd.DataFrame(data = Xvar, columns = feature_names)
        datavar = pd.concat((datavar,y),axis = 1)
        
        descvar = datavar.describe(include = "all")
        
        "Simple visualization with histogram - 10 random features"
        datahs = pd.DataFrame(data = Xvar, columns = feature_names)
        #randints = np.random.random_integers(low = 0,high = datahs.shape[1]-1,size = [1,10])
        #for r in randints[0,:]:
        #    class_histogram(datahs.iloc[:,r],y,['Control','Migraine'],'histogram')
        #    class_histogram(datahs.iloc[:,r],y,['Control','Migraine'],'boxplot')
          
        "/////////////////////////////SCALING/////////////////////////////////////"
        sc = StandardScaler()
        Xsc = sc.fit_transform(Xvar)
        datasc = pd.DataFrame(Xsc,columns = feature_names)
        datasc['class'] = y
        
        desc = datasc.describe(include = "all")
        
        "/////////////////////////////P-VALUE THRESHOLD///////////////////////////"
        "P-Value Threshold"
        class_name = 'class'
           
        datafin,colnames,_ = pvalFilter(datasc,class_name,statTests = statTests,
                       signifThresh = signifThresh,pvalSuccessRatio = pvalSuccessRatio,
                        ImbRatio = ImbRatio,min_obs = 20)
    
        descfin = datafin.describe(include = "all")
    
        Xfin = datafin.iloc[:,:-1]
        y = datafin['class']
        feature_names = colnames[:-1]
        
        
        "////////////////////P-VALUE FILTERS FOR COMORBIDITIES////////////////////"
    
        """P-Value Filter for other Comorbidities: Step 1:
        Effect of Comorbidities inside Migraine Group 
        and Control Group (separately)
        """
        
        "Necessary Initializations"
        colnames = datafin.columns #Take variable names
        rowsClass0 = castToList(np.where(datafin['class'] == 0))[0] #Control indices
        rowsClass1= castToList(np.where(datafin['class'] == 1))[0] #Migraine indices
        
        comorbidTemp = []
        for comb in comorbid:
            comorbidTemp.append(comb)
        comorbidTemp.remove('MIG')
        
        init = np.zeros((len(comorbidTemp),len(colnames) - 1)) #Initialize a counter for each variable
        counterDf = pd.DataFrame(data = init,index = comorbidTemp,columns = colnames[:-1]) #Construct DF for counters
        
        pvFCMRBStep1 = {com: {} for com in comorbidTemp} #Dictionary to save 
                                #features selected for each comorbidity
        
        
        for com in comorbidTemp:
            innerdict = {'control': [],
                        'migraine': [] }
            new_class = classes[com] #Take labels of comorbidity
            new_class_name = pd.DataFrame(classes[com]).columns[0]
            "check in controls"
            controlData = datafin.iloc[:,:-1].iloc[rowsClass0] #Remove old class
            controlData = pd.concat((controlData,new_class[rowsClass0]),axis = 1) #Append new class
            #Apply filter, return variable names that pass the filter, and binary mask
            _,colnamesCont,pvalMaskCont = pvalFilter(controlData,new_class_name,statTests = statTests,
                       signifThresh = signifThresh,pvalSuccessRatio = pvalSuccessRatioStep1,
                       ImbRatio = ImbRatio,min_obs = 20)
            
            innerdict['control'] = colnamesCont[:-1] #Save names to dictionary
            for colCont in colnamesCont[:-1]: #Increase counter
                counterDf.loc[com,colCont] = counterDf.loc[com,colCont] + 1
              
            
            "check in migraines"
            migData = datafin.iloc[:,:-1].iloc[rowsClass1]#Remove old class
            migData = pd.concat((migData,new_class[rowsClass1]),axis = 1)#Append new class
            #Apply filter, return variable names that pass the filter, and binary mask
            _,colnamesMig,pvalMaskMig = pvalFilter(migData,new_class_name,statTests = statTests,
                       signifThresh = signifThresh,pvalSuccessRatio = pvalSuccessRatioStep1,
                       ImbRatio = ImbRatio,min_obs = 20)
            
            innerdict['migraine'] = colnamesMig[:-1]
            for colMig in colnamesMig[:-1]:
                counterDf.loc[com,colMig] = counterDf.loc[com,colMig] + 1
            pvFCMRBStep1[com] = innerdict
            
        pvFCMRBStep1.update({'counterDf': counterDf})
        
        """P-Value Filter for other Comorbidities: Step 2:
        Migraine Group is combined with (an) other Comorbidit(y)/(ies)
        Effect of remaining Comorbidities is evaluated in MIG - CMRB1 - CMRB2 - ...
        and Control Group NO MIG - NO CMRB1 - NO CMRB2 - ...
        """
        
        "First find combinations of 2 and 3 comorbidities that include Migraine"
        c2 = list(combinations(comorbid,2))
        combs2 = []
        for comb in c2:
            if 'MIG' in comb:
                comb = list(comb)
                comb.remove('MIG')
                combs2.append(comb)
        c3 = list(combinations(comorbid,3))
        combs3 = []
        for comb in c3:
            if 'MIG' in comb:
                comb = list(comb)
                comb.remove('MIG')
                combs3.append(comb)
        
        combs = combs2 + combs3
        "Create Names for each comb"
        charNames = []
        for comb in combs:
            char = ''
            for c in comb:
                char = char + '_' + c
            char = 'MIG' + char
            charNames.append(char)
        
        pvFCMRBStep2 = {cname: {} for cname in charNames} #Dictionary to save each comorbidity for each combination
        
        "Iterate combs"
        for c,comb in enumerate(combs):
            
            innerdict = {com: {} for com in comorbid } #Dictionary to save 
                                    #features selected for each comorbidity
            "Necessary Initializations"
            colnames = datafin.columns #Take variable names
            datapvStep2 = pd.concat((datafin,classes[comb]),axis = 1)
            
            "Find where MIG is 1, and then where MIG- CMRBD1 - ... is 1"
            datapvStep2Pos = datapvStep2.where(datapvStep2['class'] == 1)        
            for cmbd in comb:
                datapvStep2Pos = datapvStep2Pos.where(datapvStep2Pos[cmbd] == 1) 
            datapvStep2Pos = datapvStep2Pos.dropna(axis = 0)
            "Find where MIG is 0, and then where MIG- CMRBD1 - ... is 0"
            datapvStep2Neg = datapvStep2.where(datapvStep2['class'] == 0)        
            for cmbd in comb:
                datapvStep2Neg = datapvStep2Neg.where(datapvStep2Neg[cmbd] == 0)     
            datapvStep2Neg = datapvStep2Neg.dropna(axis = 0)
            
            rowsClass0 = datapvStep2Neg.index #Control indices
            rowsClass1= datapvStep2Pos.index #Migraine + CMRBD indices
            
            comorbidTemp = []
            for com in comorbid:
                if not com in comb:
                    comorbidTemp.append(com)
            comorbidTemp.remove('MIG')
            
            init = np.zeros((len(comorbidTemp),len(colnames) - 1)) #Initialize a counter for each variable
            counterDf = pd.DataFrame(data = init,index = comorbidTemp,columns = colnames[:-1]) #Construct DF for counters
            
            
            for com in comorbidTemp:
                innerinnerdict = {'control': [],
                            'migraine': [] }
                
                new_class = classes[com] #Take labels of comorbidity
                new_class_name = pd.DataFrame(classes[com]).columns[0]
                        
                "check in controls"
                controlData = datapvStep2Neg.iloc[:,:-(len(comb) + 1)] #Remove old classes
                controlData = pd.concat((controlData,new_class[rowsClass0]),axis = 1) #Append new class
                #Apply filter, return variable names that pass the filter, and binary mask
                _,colnamesCont,pvalMaskCont = pvalFilter(controlData,new_class_name,statTests = statTests,
                           signifThresh = signifThresh,pvalSuccessRatio = pvalSuccessRatioStep2,
                       ImbRatio = ImbRatio,min_obs = 20)
                
                innerinnerdict['control'] = colnamesCont[:-1]#Save names to dictionary
                for colCont in colnamesCont[:-1]: #Increase counter
                    counterDf.loc[com,colCont] = counterDf.loc[com,colCont] + 1
                  
                
                "check in migraines"
                migData = datapvStep2Pos.iloc[:,:-(len(comb) + 1)]#Remove old classes
                migData = pd.concat((migData,new_class[rowsClass1]),axis = 1)#Append new classes
                #Apply filter, return variable names that pass the filter, and binary mask
                _,colnamesMig,pvalMaskMig = pvalFilter(migData,new_class_name,statTests = statTests,
                           signifThresh = signifThresh,pvalSuccessRatio = pvalSuccessRatioStep2,
                       ImbRatio = ImbRatio,min_obs = 20)
            
                innerinnerdict['migraine'] = colnamesMig[:-1]#Save names to dictionary
                for colMig in colnamesMig[:-1]:
                    counterDf.loc[com,colMig] = counterDf.loc[com,colMig] + 1
                
                innerdict[com] = innerinnerdict
            
            pvFCMRBStep2[charNames[c]] = innerdict
            pvFCMRBStep2[charNames[c]].update({'counterDf': counterDf})
    
        "////////////////////END OF P-VALUE FILTERS///////////////////////////////"
        "Remove features based on p-value filtering"
        
        "Step1"
        step1Df = pvFCMRBStep1['counterDf'] 
        step1cnames = step1Df.columns
        suma = []
        keepMask = []
        for step1name in step1cnames:
            suma = sum(step1Df[step1name])
            if suma > comorbidityThresh:
                keepMask.append(False) #Don't keep 
            elif suma <= comorbidityThresh:
                keepMask.append(True) #Keep
        featToKeep = step1cnames[keepMask]
        featToKeep = featToKeep.append(pd.Index(['class']))
        dataStep1Kept = datafin[featToKeep]
        
        "Step2"
        step2keys = [*pvFCMRBStep2]
        tempDf = pvFCMRBStep2[step2keys[0]]['counterDf']
        tempcols = tempDf.columns
        keepMask = np.zeros((len(step2keys),len(tempcols)))
        #keepMask has k rows, one for each key, and n columns, one for each feature
        for k,key in enumerate(step2keys):
            step2Df = pvFCMRBStep2[key]['counterDf']
            step2cnames = step2Df.columns
            suma = []        
            for n,step2name in enumerate(step2cnames):
                suma = sum(step2Df[step2name])
                if suma > comorbidityThresh:
                    keepMask[k,n] = 0 #Don't keep 
                elif suma <= comorbidityThresh:
                    keepMask[k,n] = 1 #Keep
        
        finalkeepMask = []     
        for n in range(len(tempcols)):
            suma = sum(keepMask[:,n])
            if suma > 0:
                finalkeepMask.append(True)#Keep
            elif suma <=0:
                finalkeepMask.append(False)#Don't keep
        
        featToKeep = step2cnames[finalkeepMask]
            
        featToKeep = featToKeep.append(pd.Index(['class']))    
        dataStep2Kept = datafin[featToKeep]
        
        datafinfin,_ = keep_common_cols(dataStep2Kept,dataStep1Kept)
        descfinfin = datafinfin.describe(include = "all")
    
        colnamesfin = datafinfin.columns
        Xfinfin = datafinfin.iloc[:,:-1]
        y = datafinfin['class']
        feature_names = colnamesfin[:-1]
        
        "///////////////////////////AGE FILTERING/////////////////////////////////"
        ""
        
        classes = pd.read_csv('Classes.csv') 
        ageClass = pd.read_csv('AgeClass.csv')
        dataAge = pd.concat((Xfinfin,classes,ageClass),axis = 1)
        "Pathology Check"
        dataAge = dataAge.where(dataAge[pathology] == 0)
        
        dataAge = dataAge.dropna(axis = 0)
        
        rowsClass0 = np.where(dataAge[AgeLim] == 0)
        rowsClass1 = np.where(dataAge[AgeLim] == 1)
        rows = ["t-test","Wilcoxon"]
        results = pd.DataFrame()
        
        for fEval in feature_names:
            pvec = []
            fData = dataAge[fEval]
            featClass0 = fData.iloc[rowsClass0]   
            featClass1 = fData.iloc[rowsClass1]
            if not (np.var(featClass0) == 0 or np.var(featClass1) == 0):
                if len(featClass0)/len(featClass1) > 0.99 or len(featClass0)/len(featClass1) < 1.01:
                    "t-test"
                    [tStat,pvalt] = scipy.stats.ttest_ind(featClass1, featClass0, equal_var=True)
                else:
                    "Welch's t-test"
                    [tStatt,pvalt] = scipy.stats.ttest_ind(featClass1, featClass0, equal_var=False)
                "Wilcoxon signed-rank test"
                if len(featClass0) == len(featClass1):
                    [tStatWx,pvalWx] = scipy.stats.wilcoxon(x = featClass1, y = featClass0, zero_method='wilcox')
                else:
                    try:
                        [tStatWx,pvalWx] = scipy.stats.mannwhitneyu(featClass1, featClass0)
                    except:
                        a = 1
            else:
                pvalWx = 1 #If a feature has 0 variance, it must be removed
            
            pvec.append(pvalt); pvec.append(pvalWx)
            pvec = pd.Series(pvec)
            
            results = pd.concat((results,pvec),axis = 1)
            
        results.columns = feature_names
        results.index = rows
        ageMask = []
        for feature_name in feature_names:
            test = 0
            if results[feature_name]['t-test'] < 0.01:
                test = test + 1
            if results[feature_name]['Wilcoxon'] < 0.01:
                test = test + 1
            if test == 2: #If both tests are significant
                ageMask.append(False) #Reject feature
            else:
                ageMask.append(True) #Accept feature
        
        feature_names = feature_names[ageMask]
        datafinfin = datafinfin[feature_names]
        datafinfin = pd.concat((datafinfin,y),axis = 1)
        Xfinfin = datafinfin.iloc[:,:-1]
        y = datafinfin['class']
        colnamesfin = datafinfin.columns
        feature_names = colnamesfin[:-1]
        
        "/////////////////////////END OF AGE FILTER///////////////////////////////"
        
        "////////////////////////FEATURE SELECTION////////////////////////////////" 
        
        if len(feature_names) > n_features:
        
            from collections import OrderedDict
            models = OrderedDict()
            models['lsvc'] = SVC(kernel = "linear")
            #models['rbfsvc'] = SVC(kernel = "rbf")
            models['ridge'] = RidgeClassifier()
            #models['knn'] = KNeighborsClassifier(n_neighbors = 10)
            models['logre'] = LogisticRegression()
            #models['extraTrees'] = ExtraTreesClassifier(n_estimators = 50,criterion = "gini",
            #                max_features = "sqrt",n_jobs = -1,random_state = 0,verbose = 0)
            
            bestNames = []
            allNames = []
            
            
            "SelectKBest"
            kbestscores,kbestFeatNames,Xnewkbest = feature_selection(Xfinfin,y,feature_names,alg = 'KBest',
                        kscore_func = f_classif,ksel = n_features)
            for knam in kbestFeatNames:
                bestNames.append(knam) 
                allNames.append(knam)
            
            "Lasso CV - Only for insight"
            randNames = []
            lassoScores,lassoNames = feature_selection(Xfinfin,y,feature_names,alg = 'LassoCV',
                n_features = n_features,tol = 0.005,max_iter = 2000,cv = 6,random_state = 42)
            for lnam in lassoNames:
                randNames.append(lnam)
            
            values, counts = np.unique(randNames,return_counts=True) 
            ind = np.argpartition(-counts, kth=n_features-1)[:n_features]	
            bestlassoNames = values[ind]    
            
            for lnam in bestlassoNames:
                lnam = str(lnam)
                allNames.append(lnam)
                if lnam not in bestNames:
                    bestNames.append(lnam)
            
            
            "Recursive Feature Elimination with Cross Validation"
            modNames2 = []
            opt_feat = []
            for (_,model) in models.items():
                optimal_n_features,rfecv_names,grid_scores = feature_selection(Xfinfin,y,feature_names,alg = 'RFECV',
                        estimator = model,min_features_to_select = 2)
                opt_feat.append(optimal_n_features)
                max_score = max(grid_scores)
                if (max_score >= scoreRfecvThreshold):# and (optimal_n_features <= featRfecvThreshold):
                    for rfecvnam in rfecv_names:
                        modNames2.append(rfecvnam)
                    
            values, counts = np.unique(modNames2,return_counts=True) 
            if n_features > len(counts):  
                ind = np.argpartition(-counts, kth = len(counts)-1)[:len(counts)]	
            else:
                ind = np.argpartition(-counts, kth = n_features - 1)[:n_features]	
                
            bestrfecvNames = values[ind]        
                
            for rfecvnam in bestrfecvNames:
                rfecvnam = str(rfecvnam)
                allNames.append(rfecvnam)
                if rfecvnam not in bestNames:
                    bestNames.append(rfecvnam)
            
            "ExtraTrees"
            randtreeNames= []
            for m in range(rand):
                trees_imp,trees_names = feature_selection(Xfinfin,y,feature_names,alg = 'ExtraTrees',
                        n_estimators = 100,criterion = "gini",max_features = "sqrt",
                        n_features = n_features,random_state = m + 2)
                for treenam in trees_names:
                        randtreeNames.append(treenam)
            values, counts = np.unique(randtreeNames,return_counts=True) 
            ind = np.argpartition(-counts, kth=n_features-1)[:n_features]	
            besttreeNames = values[ind]    
            
            for treenam in besttreeNames:
                treenam = str(treenam)
                allNames.append(treenam)
                if treenam not in bestNames:
                    bestNames.append(treenam)        
            
            "Permutation Importance"
            permNames = []
            for (label,model) in models.items():
                if label != 'extraTrees':
                    model = model.fit(Xfinfin,y)
                    perm_imp_features,perm_imp_mean,perm_imp_std = feature_selection(Xfinfin,y,feature_names,alg = 'PermutationImportance', 
                                classifier = model,n_repeats = 100,random_state = 72,permMeanThreshold = permMeanThreshold)
                    
                    for permnam in perm_imp_features:
                        permNames.append(permnam)
            
            values, counts = np.unique(permNames,return_counts=True) 
            if n_features > len(counts):  
                ind = np.argpartition(-counts, kth = len(counts)-1)[:len(counts)]	
            else:
                ind = np.argpartition(-counts, kth = n_features - 1)[:n_features]	
            bestpermFeatures = values[ind]  
            
            for permnam in bestpermFeatures:
                permnam = str(permnam)
                allNames.append(permnam)
                if permnam not in bestNames:
                    bestNames.append(permnam)
            """for permnam in perm_imp_features[:n_features]:
                if permnam not in bestNames:
                    bestNames.append(permnam)"""
                    
            "/////////////////////////////////////////////////////////////////////////////"
            "/////////////////////////////////////////////////////////////////////////////"
            "Conclusion"
            
            lastNames = []
            for nam in allNames:
                nam = str(nam)
                if nam not in bestNames:
                    lastNames.append(nam)
            
            values, counts = np.unique(lastNames,return_counts=True) 
            ind = np.argpartition(-counts, kth=n_features-1)[:n_features]	
            bestlastFeatures = values[ind] 
            
            for blf in bestlastFeatures:
                blf = str(blf)
                bestNames.append(blf)
        
        else: #If len(feature_names) < n_features 
            bestNames = feature_names
            
        #class_histogram(datahs[bestNames],y,['Control','Migraine'],'boxplot')
        #class_histogram(datahs[bestNames],y,['Control','Migraine'],'histogram')
            
        final_dataset = pd.concat((data[bestNames],data['class']),axis = 1)
        new_name = name + '_' + imputation + '_imp.csv'
        new_name = os.path.join(dest_folder,new_name) 
        if save_new_dataset:
            final_dataset.to_csv(new_name,index = False)
