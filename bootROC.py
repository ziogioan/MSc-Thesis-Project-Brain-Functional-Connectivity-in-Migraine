# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:04:02 2021

Thesis Project: Classification and Characterization of the Effect of Migraine 
through Functional Connectivity Characteristics: Application to EEG 
Recordings from a Multimorbid Clinical Sample

-----------------------------------------------------------------------------------------------------------------
@Authors: Ioannis Ziogas & Charalampos Lamprou
Copyright (C) 2022 Ioannis Ziogas and Charalampos Lamprou,SPBTU,ECE,AUTh
-----------------------------------------------------------------------------------------------------------------

Description: In this script, the best performing models in terms of AUC are found
and then, for these models, the bootstrap procedure is applied to calculate
the mean ROC curve.
"""
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.metrics import roc_auc_score,roc_curve
import os

def bootstrap_roc(y_pred_proba,y_true,random_seed = 42,B = 1000,threshold_length = 100):
    """
    Parameters
    ----------
    y_pred_proba : mean probability numpy array Nx1, where N the number of patients.
    Probability is defined as the probability to belong to class 1 (for example, migraine)
    
    y_true : numpy array, Nx1, class labels of N patients.
    
    random_seed : integer, optional, the default is 42.
    
    B : bootstrap iterations, optional. The default is 1000.
    
    threshold_length : Number of points used to construct a linear space
    between 0 and 1. The default is 100. Makes no difference most of the times

    Returns
    -------
    btstrp_results : Dictionary. Contains the necessary statistical quantities 
    for plotting the ROC curve, and the AUC scores.
    
    Description: This function implements a bootstrap sampling on the probabilities 
    calculated in a classification scheme, and calculates ROC curve characteristics
    from the bootstrap samples. Upper and lower 95% mean Confidence Interval is 
    calculated for the AUC score, FPR and TPR, to obtain a statistically strong
    ROC Curve. Results are stored in a dictionary and visualization of the ROC statistics
    is implemented in another function. 
    """
    
    "Calculate original fpr,tpr and AUC"
    score_orig = roc_auc_score(y_true, y_pred_proba)
    fpr_orig,tpr_orig,_ = roc_curve(y_true,y_pred_proba,pos_label = 1)
    
    boot_fpr = []
    boot_tpr =[]
    boot_auc = []
    
    rng = np.random.RandomState(random_seed)# control reproducibility 
    for i in range(B):
        "Bootstrap by sampling with replacement on the prediction indices"
        indices = rng.randint(0, len(y_pred_proba), len(y_pred_proba))
        if len(np.unique(y_true[indices])) < 2:
            """We need at least one positive and one negative class sample for ROC AUC
            to be defined: if this is not satisfied by the selected indices, 
            reject the current bootstrap sample"""
            continue
        
        "Calculate fpr,tpr,AUC scores using the bootstrap samples"
        score = roc_auc_score(y_true[indices], np.array(y_pred_proba)[indices])
        fpr,tpr,_ = roc_curve(y_true[indices],np.array(y_pred_proba)[indices],pos_label = 1)
        boot_auc.append(score)
        boot_fpr.append(fpr)
        boot_tpr.append(tpr)
    
    "boot_fpr/tpr are lists with B elements, with each element being a fpr/tpr vector"
    "Interpolate original tpr vector to have desired length (= threshold_length)"
    fpr_mean    = np.linspace(0, 1, threshold_length)
    tpr_orig    = np.interp(fpr_mean, fpr_orig, tpr_orig)
    tpr_orig[0] = 0.0 #Set first value to be 0
    tpr_orig[-1] = 1.0 #Set last value to be 1
    interp_tprs = []
    for i in range(len(boot_fpr)):
        "Take fpr & tpr bootstrap samples"
        fpr           = np.array(boot_fpr)[i]
        tpr           = np.array(boot_tpr)[i]
        "Interpolate tpr by evaluating in the points of fpr_mean"
        "fpr are the X-coordinates and tpr the Y-coordinates"
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        
    "Compute AUC score statistics"
    mean_auc = np.mean(boot_auc,axis = 0)
    std_auc = 2*np.std(boot_auc,axis = 0)
    high_auc = np.clip(mean_auc + std_auc,0,1)
    low_auc = mean_auc - std_auc
    "Compute tpr statistics"
    tpr_mean     = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std      = 2*np.std(interp_tprs, axis=0)
    tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower    = tpr_mean-tpr_std
    
    btstrp_results = {'tpr_original':tpr_orig,
               'tpr_upper':tpr_upper,
               'tpr_lower':tpr_lower,
               'tpr_mean':tpr_mean,
               'tpr_std':tpr_std,
               'tpr_orig':tpr_orig,
               'fpr_mean':fpr_mean,
               'mean_auc':mean_auc,
               'high_auc':high_auc,
               'low_auc':low_auc}
    return btstrp_results

def plot_single_roc(results,model,name):
    """
    Parameters
    ----------
    results : Dictionary. Contains the necessary statistical quantities 
    for plotting the ROC curve, and the AUC scores.
    model : str, model name.
    name : str, figure title.

    Returns
    -------
    fig : plotly figure object.

    Description: This function receives an input dictionary of bootstrapped ROC 
    characteristics, and plots the ROC curve and its 95% mean CI. 
    """
    tpr_orig = results['tpr_original']
    tpr_upper = results['tpr_upper']
    tpr_lower = results['tpr_lower']
    fpr_mean = results['fpr_mean']
    mean_auc = results['mean_auc']
    high_auc = results['high_auc']
    low_auc = results['low_auc']
    
    c_fill      = 'rgba(52, 152, 219, 0.2)'
    c_line      = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid      = 'rgba(189, 195, 199, 0.5)'
    
    fig = go.Figure([
        go.Scatter(        
        x          = fpr_mean,
        y          = tpr_upper,
        line       = dict(color=c_line, width=1),
        hoverinfo  = "skip",
        showlegend = False,
        name       = 'upper'),
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_lower,
        fill       = 'tonexty',
        fillcolor  = c_fill,
        line       = dict(color=c_line, width=1),
        hoverinfo  = "skip",
        showlegend = False,
        name       = 'lower'),
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_orig,
        line       = dict(color=c_line_main, width=2),
        hoverinfo  = "skip",
        showlegend = True,
        name       = f'AUC: {mean_auc:.3f} [{low_auc:.3f} - {high_auc:.3f}]')
    ])
    fig.add_shape(
        type ='line', 
        line =dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        title_text = model + ' ' + name,
        title_font_size= 30,
        template    = 'plotly_white', 
        title_x     = 0.5,
        xaxis_title = "1 - Specificity",
        yaxis_title = "Sensitivity",
        width       = 800,
        height      = 800,
        legend      = dict(
            yanchor="bottom", 
            xanchor="right", 
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        scaleanchor = "x", 
        scaleratio  = 1,
        linecolor   = 'black')
    fig.update_xaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        constrain   = 'domain',
        linecolor   = 'black')
    plot(fig, auto_open = True)
    
    return fig           

"Initializations"

chooseCom = 'MIG'

pwd = os.getcwd()
res_folder = 'KFoldResults'
res_folder = os.path.join(pwd,res_folder)
inner_res_folder = 'conv'#'conv','crossBspec','crossBspec_noBic'
if not os.path.exists(res_folder):
    os.mkdir(res_folder)
res_folder = os.path.join(res_folder,inner_res_folder)
if not os.path.exists(res_folder):
    os.mkdir(res_folder)
if inner_res_folder == 'conv':
    res_folder = os.path.join(res_folder,'finalBest')

classes = pd.read_csv('Classes.csv')
Class = classes[chooseCom].values
categ = ['EO', 'EC', 'all', 'DiffRatio']


for file in os.listdir(res_folder):
    
    if not os.path.isdir(file):
        if 'selected' not in file and 'age' in file:        
            
            for cat in categ:
                name = inner_res_folder
                if cat in file:
                     name = name + '_' + cat
            
            dict1 = np.load(os.path.join(res_folder,file),allow_pickle = True).item()
            keys1 = [*dict1]
            
            topModels = []
            maxAUC = 0
            for k,key in enumerate(keys1):
                rocAUC = []
                dict2 = dict1[key]
                results = dict2['results']
                reskeys = [*results] #keys, names of classifiers
                for keykey in reskeys:
                    rocAUC.append(results[keykey]['test_AUC'][1])#AUC scores for each model
                rocAUC = np.array(rocAUC)
                ind = np.argpartition(-rocAUC, kth=3)[:3]#Top-3 performing models	
                topModels.append([reskeys[aucind] for aucind in ind])#Names
                if sum(rocAUC[ind]) > maxAUC:
                    maxAUC = sum(rocAUC[ind])
                    pos = k
            #pos = 2
            name = name + ', for k = ' + str(keys1[pos][1:])
            bestK = dict1[keys1[pos]]
            bestModels = topModels[pos]    
            for model in bestModels:
                probs = bestK['probabilities'][model]
                bootResults = bootstrap_roc(probs,Class,random_seed = 42,
                                    B = 1000,threshold_length = 100)
                
                plot_single_roc(bootResults,model,name)           
            
            l = 1
            

