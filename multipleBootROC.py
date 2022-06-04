# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:58:01 2021

Thesis Project: Classification and Characterization of the Effect of Migraine 
through Functional Connectivity Characteristics: Application to EEG 
Recordings from a Multimorbid Clinical Sample

-----------------------------------------------------------------------------------------------------------------
@Authors: Ioannis Ziogas & Charalampos Lamprou
Copyright (C) 2022 Ioannis Ziogas and Charalampos Lamprou,SPBTU,ECE,AUTh
-----------------------------------------------------------------------------------------------------------------

Description: In this script the KFold CV results are loaded (must be available) and the calculated
probability vector for the patients is used to construct a ROC curve with a bootstrap sampling of the 
probabilities indices to calculate many instances of TPR vectors. This procedure is implemented
for the best performing number of features for each FC measure and for each case (EO/EC/all) and results 
are plotted for each specified classifier in a single x-y axis system to compare between different measures
with the same classifier and in the same case. 
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.colors as matcolors
import plotly.colors as plotlycolors

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
    
    rng = np.random.RandomState(random_seed)#control reproducibility   
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
    tpr_orig[0] = 0.0
    tpr_orig[-1] = 1.0
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

def plot_roc(results,model,name,ots,plot_labels,kDict,open_plots,langswitch):
    """
    Parameters
    ----------
    results : Dictionary. Contains the necessary statistical quantities 
    for plotting the ROC curve, and the AUC scores.
    model : str, model name.
    name : str, figure title.
    ots: list of str, measures to be compared in the ROC plot (different ROC curves for each measure
                are going to be displayed in the same axis system)
    plot_labels: list of str, legend labels for each measure
    kDict: Dictionary of dictionaries. 1st level: eye case (EO/EC/all). 2nd level: FC measure (COH,PAC,PLV,crossBspec_anti).
    Each combination of keys is mapped to a number of best selected features 'k'. 
    open_plots: boolean flag. Flag to open plots (1 to open, 0 to flag).
    langswitch: str. Determines language of plots ('eng' or 'gr').
    
    Returns
    -------
    fig : plotly figure object.

    Description: This function receives an input dictionary of bootstrapped ROC 
    characteristics, and plots the ROC curve and its 95% mean CI. 
    """
    c_grid = 'rgba(189, 195, 199, 0.5)'
    fig = go.Figure()
    fig.update_layout(title_text = name + ' ' + model,
            title_font_size= 40)
        
    for o,ot in enumerate(ots):
        tpr_orig = results[ot]['tpr_original']
        tpr_upper = results[ot]['tpr_upper']
        tpr_lower = results[ot]['tpr_lower']
        fpr_mean = results['fpr_mean']
        mean_auc = results[ot]['mean_auc']
        high_auc = results[ot]['high_auc']
        low_auc = results[ot]['low_auc']
        c_fill = results[ot]['c_fill']
        c_line = results[ot]['c_line']
        c_line_main = results[ot]['c_line_main']        
        ksel = kDict[ot]
        
        "x- and y-labels in english or in greek"
        if langswitch == 'gr':
            xaxtit = "1 - Εξειδίκευση"
            yaxtit = "Ευαισθησία"
            legtex1 = ''#', για k = '+ str(ksel)
            legtex2 = 'ΕΚΚ'
        elif langswitch == 'eng':
            xaxtit = "1 - Specificity"
            yaxtit = "Sensitivity"
            legtex1 = ', for k = '+ str(ksel) 
            legtex2 = 'AUC'
        
        
        
        fig.add_scatter(
            x          = fpr_mean,
            y          = tpr_upper,
            line       = dict(color=c_line, width=1),
            hoverinfo  = "skip",
            showlegend = False,
            name       = 'upper')
        fig.add_scatter(
            x          = fpr_mean,
            y          = tpr_lower,
            fill       = 'tonexty',
            fillcolor  = c_fill,
            line       = dict(color=c_line, width=1),
            hoverinfo  = "skip",
            showlegend = False,
            name       = 'lower')
        fig.add_scatter(
            x          = fpr_mean,
            y          = tpr_orig,
            line       = dict(color=c_line_main, width=2),
            hoverinfo  = "skip",
            showlegend = True,
            name       = plot_labels[o] + legtex1 + ' - ' + legtex2+f': {mean_auc:.3f} [{low_auc:.3f} - {high_auc:.3f}]')
        fig.add_shape(
            type ='line', 
            line =dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            template    = 'plotly_white', 
            title_x     = 0.5,
            xaxis_title = xaxtit,
            yaxis_title = yaxtit,
            width       = 1200,
            height      = 1200,
            legend      = dict(
                font = dict(size = 33),
                yanchor="bottom", 
                xanchor="right", 
                x=0.95,
                y=0.01,
            ),
            font = dict(size = 25)
        )
        fig.update_yaxes(
            range       = [0, 1],
            gridcolor   = c_grid,
            scaleanchor = "x", 
            scaleratio  = 1,
            linecolor   = 'black'
            #rangeselector = dict(font = dict(size = 25))
            )
        fig.update_xaxes(
            range       = [0, 1],
            gridcolor   = c_grid,
            constrain   = 'domain',
            linecolor   = 'black'
            #rangeselector = dict(font = dict(size = 25))
            )
    
    plot(fig, auto_open = open_plots)
    
    return fig

def make_colors(color_name,opacity):
    """
    Parameters
    ----------
    color_name : str, name of color (e.g. 'green').
    opacity : float. Ranges from 0 to 1, determines opacity.

    Returns
    -------
    rgbastr : str list containing rgb triplet and opacity value (output example: rgba(0,128,0,1) -> green, op. = 1).

    Description: This function receives as input a desired color's name and an opacity value and returns a rgba quadruplet
    """
    triplet = matcolors.to_rgb(color_name)
    triplet = plotlycolors.convert_to_RGB_255(triplet)
    
    rgba = list(triplet)
    rgba.append(opacity)
    
    rgbastr = [str(el) for el in rgba]
    #Construct plotly color format
    rgbastr = 'rgba(' + ",".join(rgbastr) + ')'
    
    return rgbastr    

"Initializations"

chooseCom = 'MIG' #Class of interest
pwd = os.getcwd()
res_folder1 = 'KFoldResults'
res_folder1 = os.path.join(pwd,res_folder1)
res_folder2 = 'KFoldResults_MI'
res_folder2 = os.path.join(pwd,res_folder2)
dest_folder1 = os.path.join(res_folder1,'methodComparisonBest')
if not os.path.exists(dest_folder1):
    os.mkdir(dest_folder1)
dest_folder_gr = os.path.join(pwd,'thesis_greek_res')
if not os.path.exists(dest_folder_gr):
    os.mkdir(dest_folder_gr)
dest_folder_gr = os.path.join(dest_folder_gr,'methodComparisonROC_bigger_legend')
if not os.path.exists(dest_folder_gr):
    os.mkdir(dest_folder_gr)

"Load necessary classes table (contains class vector for each psychopathology"
classes = pd.read_csv('Classes.csv')
Class = classes[chooseCom].values #Keep only class of interest (migraine)
categories = ['EC','EO','all']#]'DiffRatio',,
save_plots = False #True to save plots
open_plots = True # True to open plots
langswitch = 'gr' #Language to be displayed in the ROC graph'gr'/'eng'
typ = 'all' #To be included in the saved .jpg filename
"Plot titles in english and in greek"
if langswitch == 'gr':
    titleName = 'Σύγκριση Απόδοσης με την ΚΧΛΔ' 
elif langswitch == 'eng':
    titleName = 'Connectivity Measures Comparison' 
"List of classifier names"
models = ["SVC","LinearSVC","kNN","LogRe","DecisionTree","GaussianProcess"]#,"NaiveBayes","DecisionTree"]
#methods = ['COH','crossBspec','PAC']
#methods = ['COH','PLV','PAC_tf']
"List of FC measures names"
methods = ['COH','PLV','crossBspec_anti','PAC']
#plot_labels = ['Coherence','cross-Bispectrum','SwD-PAC']
#plot_labels = ['Coherence','PLV','PAC_tf']
if langswitch == 'gr':
    plot_labels = ['ΦΣ','ΔΚΦ','ΑΔΕΦ/ΑΔΕΣ','ΔΔ']
elif langswitch == 'eng':
    plot_labels = ['COH','PLV','crossBspec_anti','PAC']

"""Construct list of strings with names of the fields that the dictionary that stores 
results from the bootstrap ROC procedure will have."""
fields = ['tpr_original','tpr_upper','tpr_lower','mean_auc','high_auc',
          'low_auc','c_fill','c_line','c_line_main']

"Translations from english to greek"
EngToGrMetrics = {'COH':'ΦΣ','PLV':'ΔΚΦ','PAC':'ΔΔ', 'crossBspec_anti':'ΑΔΕΦ/ΑΔΕΣ'}
EngToGrModels = {'SVC':'ΜΓΜΔΥ','LinearSVC':'ΓΜΔΥ','kNN':'κΠΓ',
                 'LogRe':'ΛΠ', 'GaussianProcess':'ΓΔ','DecisionTree':'ΔΑ'}
EngToGrCategs = {'EC':'ΜΚ','EO':'ΜΑ','all':'Συνδ. Όλων'}


"Define colors for ROC plot"

#Blue - Coherence
c_fillCOH      = make_colors('blue',0.35) #blue: #0000FF
c_lineCOH     = make_colors('blue',0.5)
c_line_mainCOH = make_colors('blue',1)

#Red - crossBispectrum
c_fillCB      = make_colors('red',0.35)#red: #FF0000
c_lineCB     = make_colors('red',0.5)
c_line_mainCB = make_colors('red',1)

#Green - SwD-PAC 
c_fillPAC = make_colors('green',0.35) #green: #15B01A    
c_linePAC = make_colors('green',0.5)
c_line_mainPAC = make_colors('green',1)


#Yellow - PLV
c_fillPLV = make_colors('yellow',0.35)
c_linePLV = make_colors('yellow',0.5)
c_line_mainPLV = make_colors('yellow',1)

colors = {'COH': {'c_fill':c_fillCOH, 'c_line':c_lineCOH,'c_line_main':c_line_mainCOH}, 
          'crossBspec_anti': {'c_fill':c_fillCB, 'c_line':c_lineCB,'c_line_main':c_line_mainCB},
          'PAC': {'c_fill':c_fillPAC, 'c_line':c_linePAC,'c_line_main':c_line_mainPAC},
          'PLV': {'c_fill':c_fillPLV, 'c_line':c_linePLV,'c_line_main':c_line_mainPLV},
          'PAC_tf': {'c_fill':c_fillCOH, 'c_line':c_lineCOH,'c_line_main':c_line_mainCOH}}
kdict = {cat: {method: [] for method in methods} for cat in categories}

"Determine kvalues for which the bootstrap and plotting procedure should be implemented"
kdict['EO']['COH'] = 21; kdict['EC']['COH'] = 11; kdict['all']['COH'] = 36; 
kdict['EO']['crossBspec_anti'] = 46; kdict['EC']['crossBspec_anti'] = 51; kdict['all']['crossBspec_anti'] = 50;  #5,40,30
kdict['EO']['PAC'] = 16; kdict['EC']['PAC'] = 6; kdict['all']['PAC'] = 50; 
kdict['EO']['PLV'] = 10; kdict['EC']['PLV'] = 15; kdict['all']['PLV'] = 28; 
#kdict['EO']['PAC_tf'] = 30; kdict['EC']['PAC_tf'] = 12; kdict['all']['PAC_tf'] = 50; 

"""
kdict['EO']['COH'] = 10; kdict['EC']['COH'] = 10; kdict['all']['COH'] = 30; 
kdict['EO']['crossBspec'] = 10; kdict['EC']['crossBspec'] = 10; kdict['all']['crossBspec'] = 30; 
kdict['EO']['PAC'] = 10; kdict['EC']['PAC'] = 6; kdict['all']['PAC'] = 30; 
"""

"Implement the procedure for each eye case and for each classifier"
for cat in categories:
            
    for model in models:
        all_results = {}
        bestKdict = {method: [] for method in methods}
        
        if langswitch == 'gr':
            
            modelname = EngToGrModels[model]
            
            catname = EngToGrCategs[cat]
        
        elif langswitch == 'eng':
            
            modelname = model
            
            catname = cat
            
        "Implement bootstrap for each FC measure"
        for method in methods:
                
            
            inner_res_folder = method #'COH','PLV','crossBspec_anti','PAC'
            if inner_res_folder == 'crossBspec_anti':
                method_folder = os.path.join(res_folder2,inner_res_folder)
            else:
                method_folder = os.path.join(res_folder1,inner_res_folder)
                
            file = method + '_' + cat + '_Dataset.npy'
            
            methDict = np.load(os.path.join(method_folder,file),allow_pickle = True).item()
            bestK = methDict['k'+str(kdict[cat][method])]
            bestKdict[method] = str(kdict[cat][method])
        
            all_results[method] = {f:[] for f in fields}
            probs = bestK['probabilities'][model]
            "Bootstrap procedure"
            bootResults = bootstrap_roc(probs,Class,random_seed = 42,
                                    B = 1000,threshold_length = 100)
            

            all_results[method]['tpr_original'] = bootResults['tpr_original']
            all_results[method]['tpr_upper'] = bootResults['tpr_upper']
            all_results[method]['tpr_lower'] = bootResults['tpr_lower']
            all_results[method]['mean_auc'] = bootResults['mean_auc']
            all_results[method]['high_auc'] = bootResults['high_auc']
            all_results[method]['low_auc'] = bootResults['low_auc']
            all_results['fpr_mean'] = bootResults['fpr_mean']
            all_results[method]['c_fill'] = colors[method]['c_fill']
            all_results[method]['c_line'] = colors[method]['c_line']
            all_results[method]['c_line_main'] = colors[method]['c_line_main']
        
        "Plotting procedure"
        fig = plot_roc(all_results,modelname,titleName + ' - ' + catname+':',methods,plot_labels,bestKdict,open_plots,langswitch)
        
        "Save"
        if save_plots:
            if langswitch == 'gr':
                img_path = os.path.join(dest_folder_gr,cat + '_' + model)
                fig.write_image(img_path + typ + ".png", engine = "orca")
            elif langswitch == 'eng':
                img_path = os.path.join(dest_folder1,cat + '_' + model)
                fig.write_image(img_path + typ + ".png", engine = "orca")
