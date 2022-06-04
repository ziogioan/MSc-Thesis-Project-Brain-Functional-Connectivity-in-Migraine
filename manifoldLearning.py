# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:11:10 2022

Thesis Project: Classification and Characterization of the Effect of Migraine 
through Functional Connectivity Characteristics: Application to EEG 
Recordings from a Multimorbid Clinical Sample

-----------------------------------------------------------------------------------------------------------------
@Authors: Ioannis Ziogas & Charalampos Lamprou
Copyright (C) 2022 Ioannis Ziogas and Charalampos Lamprou,SPBTU,ECE,AUTh
-----------------------------------------------------------------------------------------------------------------

Description: In this script, datasets after the feature selection procedure are loaded,
and different manifold (dimensionality reduction) algorithms are utilized to 
provide low-dimensional representations of the initial high-dimensional feature space.
The results of each algorithm are plotted with different colors for each class,
to achieve a rough visualization of the positioning of data points of the two classes
in the high-dimensional space.
"""

import pandas as pd
import numpy as np
import numpy.matlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from collections import OrderedDict
from functools import partial
import os
import random


def pca_vis_3(df,leg_classes,X,num_colors,plot_flag):
     
     """High Dimensional Visualization - basic PCA - required for t-SNE if input 
     dataset is high-dimensional (more than 50 dimensions).
     PCA with 3 components, and 2d - 3d visualization
     X must be np array, (n_samples,n_features) - no class"""
     
     pca = PCA(n_components=3)
     Xpca = pca.fit_transform(X)
     dfmed = df
     dfmed['pca-one'] = Xpca[:,0]
     dfmed['pca-two']= Xpca[:,1]
     dfmed['pca-three']= Xpca[:,2]
     
     classvec = df["class"]
     classvec.where(classvec == 0,other = leg_classes[1],inplace = True)
     classvec.where(classvec == leg_classes[1], other = leg_classes[0], inplace = True)
     
     
     exp_var = pca.explained_variance_ratio_
     exp_var_tot3 = sum(exp_var)
 
     exp_var_tot2 = sum(exp_var[:-1])
     if plot_flag:
         plt.figure(figsize = (16,10),dpi = 350)
         sns.scatterplot(x= "pca-one",y = "pca-two",
                     hue = "class",
                     palette=sns.color_palette("hls", num_colors),
                     data=df,
                     legend="full",
                     alpha=1)
         plt.title('2 PCs, ' + name + ' ' + ' Explained Variance: ' 
                   + str(round(100*exp_var_tot2,2)) + ' %')
         if save_plots:
             plt.savefig(os.path.join(case_folder,'2PCs.png'))
         
         classvec.where(classvec == leg_classes[1],other = 0,inplace = True)
         classvec.where(classvec == 0, other = 1, inplace = True)
 
         
         ax = plt.figure(figsize=(16,10)).gca(projection='3d')
         ax.scatter(
             xs=dfmed["pca-one"], 
             ys=dfmed["pca-two"], 
             zs=dfmed["pca-three"], 
             c=dfmed["class"]
         )
         ax.set_xlabel('pca-one')
         ax.set_ylabel('pca-two')
         ax.set_zlabel('pca-three')
         plt.show()
         
     return dfmed,exp_var_tot3,exp_var_tot2,exp_var    

def pca_gen(X,n_comp,svd):
    """
    Description: This function applies the PCA algorithm to perform a dimensionality reduction
    to the input dataset. 
    Parameters
    ----------
    X : array-like, input dataset.
    n_comp : number of principal components.
    svd : solver.

    Returns
    -------
    Xpca : output dataset after transformation.
    exp_var_tot : total explained variance (sum of explained variances of all components).
    exp_var : vector of explained variances for each component. 
    """
    
    if svd == 'full':
        pca = PCA(n_components = n_comp,svd_solver = svd)
    else:
        pca = PCA(n_components = n_comp)
    Xpca = pca.fit_transform(X)
    exp_var = pca.explained_variance_ratio_
    exp_var_tot = sum(exp_var)

    return Xpca,exp_var_tot,exp_var

def tsne_search(X,n_components,start,end,n_iter,method,random = 5,penalty = 0.02):
    """
    Description: This function performs a grid search to optimize the parameters
    that will be fed in the t-SNE algorithm to calculate a low-dimensional representation
    of the input data X

    Parameters
    ----------
    X : pandas (N subjects x M variables) matrix.
    n_components : int. Dimension of the embedded space.
    start : int. Lower limit of perplexity values to test.
    end : int. Upper limit of perplexity values to test.
    n_iter : int. Number of iterations for the optimization process.
    method : str, 'penalty'/perplexity. Method for calculating a score for each random state
    and perplexity value. Imposes a different perplexity-based penalty on the score  
    random : int. Number of random shuffles, optional. The default is 5.
    penalty : float. Penalty that is taken into account in the scoring procedure of
    each manifold representation, optional. The default is 0.02.

    Returns
    -------
    opt_perplex : Optimal perplexity value based on the best score.
    opt_seed : Optimal random initialization based on the best score.
    kl_min_score : best score. Kullback-Leibler divergence or a perplexity based score.    
    """
    
    score = np.zeros((end+1-start,random))
    for i in range(start,end+1):
        for r in range(0,random):
            tsne = manifold.TSNE(n_components=n_components, verbose=0,
            perplexity=i, n_iter = n_iter,init = 'pca',random_state = r)
            tsne_results = tsne.fit_transform(X)
            if method == "penalty":
                score[i-start,r] = (tsne.kl_divergence_) + i*penalty
            elif method == "perplexity":
                score[i-start,r] = 1/i + i/X.shape[0]

 
    kl_min_score = np.min(score)
    inds = np.where(score == kl_min_score)
    if len(inds[0])==1:
        opt_perplex = inds[0].item() + start
        opt_seed = inds[1].item() 
    else:
        opt_perplex = inds[0][0].item() + start
        opt_seed = inds[1][0].item()

    return opt_perplex,opt_seed,kl_min_score

def grid_search(algorithm,X,n_components,n_neighbors,score_name,limit = 5, **kwargs):
    """
    Description: This function implements a grid search for hyper-parameter tuning. It can be applied
    to most of the manifold learning algorithms of sklearn. If a score measure is available,
    the most suitable number of neighbours value is determined by finding the minimum score 
    
    Parameters
    ----------
    algorithm : object, sklearn.manifold.ALG.
    X : pandas dataframe, (N subjects x M variables) matrix..
    n_components : low dimension of representation.
    n_neighbors : int. Number of neighbors, necessary for most algorithms.
    score_name : str, name of the scoring measure of an algorithm. Not all algorithms have one.
    limit : int, optional, the default is 5. Up and down offset of the possible values set for the number of neighbors.
    The grid search is implemented over the number of neighbours and the scoring function. 
    **kwargs : additional arguments, that are not common to all algorithms.

    Returns
    -------
    ind + lower: optimal number of neighbours.
    Y : array-like, (N subjects x M columns, M = n_components) low-dimensional representation.
    """
    
    algorithm.n_components = n_components
    upper = n_neighbors + limit
    lower = n_neighbors - limit
    Y = []
    for key, value in kwargs.items():
        algorithm.key = value
    if lower < 5:
        lower = 5
    score = np.zeros((1,len([*range(lower,upper)])))
    for i in range(lower,upper):
        algorithm.n_neighbors = i
        Y.append(algorithm.fit_transform(X))
        if hasattr(algorithm, score_name):
            score[0,i-lower] = getattr(algorithm,score_name)
        else:
            score[0,i-lower] = algorithm.reconstruction_error()
    min_score = np.min(score)
    ind = np.where(score == min_score)[1].item()
    Y = Y[ind]
    return ind + lower, Y

"Initializations"
sns.set_theme() #Set seaborn background for figures
path_init = os.getcwd()#Thesis Python
"case: determine which data points will be used for the dimensionality reduction - of both classes or of a single"
case = 'both_classes' #positive_class, negative_class, both_classes
cats = ['all'] #Determine eye case ['all','EO','EC']"
feats = ['PAC'] #Determine FC measure ['COH','PLV','PAC']#'crossBspec_anti'"
save_plots = True #True to save plots
"howmany: determine if a single or many algorithms will be used (affects only the plotting procedure)"
howmany = 'single' #'many'
langswitch = 'gr' #Determine language 'gr/'eng'"
n_components = 2
method = "perplexity" #method to find the optimal perplexity
save_tsne = True

"Translations from english to greek"
EngToGrMetrics = {'COH':'ΦΣ','PLV':'ΔΚΦ','PAC':'ΔΔ', 'crossBspec_anti':'ΑΔΕΦ/ΑΔΕΣ'}
EngToGrCategs = {'EC':'ΜΚ','EO':'ΜΑ','all':'Συνδυασμός Όλων'}
EngToGrAlgs = {'t-SNE':'t-ΣΕΓ','SE':'ΦΕ','Isomap':'ΙΧΧ','MDS':'ΠΔΚ'}

"Repeat manifold learning for each eye case and for each FC measure"
for cat in cats:
    for feat in feats:
        "Find data path"
        if feat == 'crossBspec_anti':
            newSets_folder = os.path.join(path_init,'newDatasets_MI')
        else:
            newSets_folder = os.path.join(path_init,'newDatasets')
        setsCat_folder = os.path.join(newSets_folder,feat)
        
        "Make filename"
        name = feat + '_' + cat + '_Dataset'
        
        "Translate plot text if needed"
        if langswitch == 'gr':
            legendClasses = ['ΑχΗ','ΑμΗ']
            metricname = EngToGrMetrics[feat]
            catname = EngToGrCategs[cat]
        elif langswitch == 'eng':
            legendClasses = ['Control','Migraine']
            metricname = feat
            catname = cat
        
        "Different save path for each language"
        if langswitch == 'eng':
            datasetSpec_folder = os.path.join('tsne',name)
            if not os.path.exists(datasetSpec_folder):
                os.mkdir(datasetSpec_folder)
                
            case_folder = os.path.join(datasetSpec_folder,case)
            if not os.path.exists(case_folder):
                os.mkdir(case_folder)
        
        elif langswitch == 'gr':
            path_dest = os.path.join(path_init,'thesis_greek_res','manifold')
            if not os.path.exists(path_dest):
                os.mkdir(path_dest)
            datasetSpec_folder = os.path.join(path_dest,name)
            if not os.path.exists(datasetSpec_folder):
                os.mkdir(datasetSpec_folder)
                
            case_folder = os.path.join(datasetSpec_folder,case)
            if not os.path.exists(case_folder):
                os.mkdir(case_folder)
            
        "Read data"
        dataset = pd.read_csv(os.path.join(setsCat_folder,name + '.csv'))
        data1 = dataset[dataset['class'] == 1]
        data0 = dataset[dataset['class'] == 0]
        
        #desc1 = data1.describe()
        #desc0 = data0.describe()
        
        if case == 'both_classes':
            X = dataset.iloc[:,:-1]
            y = dataset.iloc[:,-1]
            num_colors = 2
        elif case == 'positive_class':
            X = data1.iloc[:,:-1]
            y = data1.iloc[:,-1]
            num_colors = 1
        elif case == 'negative_class':
            X = data0.iloc[:,:-1]
            y = data0.iloc[:,-1]
            num_colors = 1
        
        "Standard scaling of data"
        sc = StandardScaler()
        Xsc = sc.fit_transform(X.values)
        
        "Make scaled dataset"
        feat_cols = dataset.iloc[:,:-1].columns
        datasc = pd.DataFrame(Xsc,columns = feat_cols)
        datasc['class'] = y
        
        "Make plot colors according to class label of each subject"
        targets = datasc['class'].astype('category')
        label_color = ['red' if i==1 else 'blue' for i in targets]
        
        "/////////////////////////////////////////////////////////////////////////////"
        
        "/////////////////////////////////////////////////////////////////////////////"
        
        "A simple PCA visualization"
        data3d,exp_var_tot3,exp_var_tot2,_ = pca_vis_3(datasc,legendClasses,Xsc,num_colors,plot_flag = True)
        
        "If number of features is bigger than 50 use PCA for dimentionality reduction"
        if datasc.shape[1] > 50:
            pca_comp = 30
            Xpca,exp_var_tot,_ = pca_gen(Xsc,pca_comp,svd = 'notfull')
        else:
            Xpca = Xsc
        
        "/////////////////////////////////////////////////////////////////////////////"
            
        "Comparison of Manifold Learning Techniques"
        
        """Y: Dictionary in which the low-dimensional representations from each algorithm 
        are stored"""
        Y = OrderedDict()
        """opt_nei: Dictionary in which the optimal number of neighbours value from each algorithm 
        is stored"""
        opt_nei = OrderedDict()
        
        "/////////////////////////////////////////////////////////////////////////////"
        
        "Check if the parameters for t-SNE are already saved from a previous run. Else run t-SNE"
        tsne_folder = os.path.join(path_init,'tsne')
        tsne_file = os.path.join(tsne_folder,name+'.npy')
        if not os.path.exists(tsne_file):
            opt_perplex,opt_seed,kl_score = tsne_search(Xpca,n_components = 2,
                start = 5,end = 30,n_iter = 300,method = "perplexity",random = 5,penalty = 0.015)
            tsnedict = {'perplexity':opt_perplex,
                        'random seed':opt_seed,
                        'kl score':kl_score}
            
            if save_tsne:
                if not os.path.exists(tsne_folder):
                    os.mkdir(tsne_folder)
                np.save(tsne_file,tsnedict)
        else:
            tsne_dict = np.load(tsne_file,allow_pickle='TRUE').item()
            opt_perplex = tsne_dict['perplexity'] 
            opt_seed = tsne_dict['random seed'] 
            kl_min_score = tsne_dict['kl score'] 
        
        """For comparison and precision reasons, the number of neighbours central value of the search interval for all other 
        algorithms, is set equal to the optimal number of neighbours as came out of t-SNE"""
        
        "Optimal number of neighbors came out from t-SNE grid search"
        n_neighbors = opt_perplex
        
        "t-distributed Stochastic Neighbor Embedding"
        tSNE = manifold.TSNE(n_components = 2, verbose = 0,
                perplexity = opt_perplex, n_iter = 300,init = 'pca',random_state = opt_seed)
        #Y['t-SNE'] = tSNE.fit_transform(Xpca)
        #opt_nei['t-SNE'] = n_neighbors
        
        "Locally Linear Embedding"
        #LLE = manifold.LocallyLinearEmbedding(eigen_solver = 'auto')
        
        "Standard LLE"
        #opt_nei['LLE'],Y['LLE'] = grid_search(LLE,Xsc,n_components,n_neighbors,"reconstruction_error_",
        #                                      limit = 1, method = 'standard')
        """
        "Hessian LLE  n_neighbors > n_components*(n_components + 3)/2"
        opt_nei['Hessian LLE'],Y['Hessian LLE'] = grid_search(LLE,Xsc,n_components,n_neighbors,
                                                             "reconstruction_error_",limit = 5, method = 'hessian')
        
        
        "Modified LLE"
        opt_nei['Modified LLE'], Y['Modified LLE'] = grid_search(LLE,Xsc,n_components,n_neighbors,
                                                                "reconstruction_error_",limit = 5, method = 'modified') 
        
        "Local Tangent Space Alignment"
        opt_nei['LTSA'],Y['LTSA'] = grid_search(LLE,Xsc,n_components,n_neighbors,
                                                "reconstruction_error_",limit = 5, method = 'ltsa')
        """
        
        "Spectral Embedding"
        SE = manifold.SpectralEmbedding(n_components = n_components,n_neighbors = n_neighbors)
        Y['SE'] = SE.fit_transform(Xsc)
        opt_nei['SE'] = n_neighbors
         
        "ISOMAP"
        Isomap = manifold.Isomap()
        #opt_nei['Isomap'],Y['Isomap'] = grid_search(Isomap,Xsc,n_components,n_neighbors,
        #                                          "reconstruction_error_",limit = 5)
        
        "Multi-dimensional Scaling"
        MDS = manifold.MDS(max_iter = 300,n_init = 2) 
        #opt_nei['MDS'],Y['MDS'] = grid_search(MDS,Xsc,n_components,n_neighbors,"stress_",limit = 5)
        
        "Random Trees Embedding"
        #RTE = make_pipeline(RandomTreesEmbedding(n_estimators = 200,
        #           max_depth = 5,random_state = 0),
        #          TruncatedSVD(n_components = 2))
        #Y['RTE'] = RTE.fit_transform(Xsc,y)
        
        
        "Truncated SVD"
        #TSVD = TruncatedSVD(n_components = 2)
        #Y['TSVD'] = TSVD.fit_transform(Xsc,y)
        
        """
        "Sparse Random Projection"
        SRP = SparseRandomProjection(n_components=2, random_state=42)
        Y['SRP'] = SRP.fit_transform(Xsc,y)
        """
        
        "/////////////////////////////////////////////////////////////////////////////"
        "/////////////////////////////////////////////////////////////////////////////"
        "SUPERVISED ALGORITHMS"
        
        "Neighborhood Component Analysis"
        #NCA = NeighborhoodComponentsAnalysis(n_components = 2,
        #        init = "random",random_state = 0)
        #Y['NCA'] = NCA.fit_transform(Xsc,y)
        
        
        "/////////////////////////////////////////////////////////////////////////////"
        
        "Plot results"
        
        "Make plot"

        """ax  = fig.add_subplot(251,projection = '3d')
        ax.scatter(data3d['pca-one'],data3d['pca-two'],data3d['pca-three'],c = label_color,
                   cmap = plt.cm.Spectral) 
        ax.view_init(4,-72) #for 3d only
        ax.set_title("3D-PCA, %f percent explained variance"
                          %(100*exp_var_tot3))"""
        
        "Translate suptitle of figure if necessary"
        if langswitch == 'gr':
            suptittex = 'Μείωση Διάστασης: ' + metricname + ' - ' + catname
        elif langswitch == 'eng':
            suptittex = 'Manifold Learning: ' + metricname + ' - ' + catname
            
        "Follow a different procedure for a single algorithm and for many algorithms"
        if howmany == 'single':
            label = [*Y][0]
            y = Y[label]
            if langswitch == 'gr':
                algname = EngToGrAlgs[label]
            elif langswitch == 'eng':
                algname = label
            ax = plt.figure(figsize = (22,13))
            plt.suptitle(suptittex,fontsize = 35)
            plt.scatter(y[:,0],y[:,1], c = label_color,cmap = plt.cm.Spectral)
            #ax.set_title("%s" %(label))
            #plt.title('Μείωση Διάστασης μέσω '+ algname+' : ' + metricname + ' - ' + catname)
            plt.title(algname,fontsize = 23)
            plt.xlabel('y1',fontsize=18)
            plt.ylabel('y2',fontsize =18)    
            plt.xticks(color = 'w')
            plt.yticks(color = 'w')
            plt.axis('tight')
            plt.legend([legendClasses[1],legendClasses[0]],fontsize = 20)
            if save_plots:
                if not os.path.exists(os.path.join(case_folder,howmany)):
                    os.mkdir(os.path.join(case_folder,howmany))
                plt.savefig(os.path.join(case_folder,howmany,feat + '_' + cat+'_'+algname+'_manifold.png'),bbox_inches='tight',dpi = 300)
                #plt.savefig(os.path.join(case_folder,'manifold2.png'))
            else:
                plt.show()
        elif howmany == 'more':
            fig = plt.figure(figsize = (22,13))
            for i, (label,y) in enumerate(Y.items()): 
                if langswitch == 'gr':
                    algname = EngToGrAlgs[label]
                elif langswitch == 'eng':
                    algname = label
                #ax = plt.figure()
                ax = fig.add_subplot(2,2,1 + i)# + (i>3))
                plt.scatter(y[:,0],y[:,1], c = label_color,cmap = plt.cm.Spectral)
                #ax.set_title("%s" %(label))
                #plt.title('Μείωση Διάστασης μέσω '+ algname+' : ' + metricname + ' - ' + catname)
                ax.set_title(algname,fontsize = 23)
                ax.set_xlabel('y1',fontsize=18)
                ax.set_ylabel('y2',fontsize =18)    
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.axis('tight')
                plt.legend([legendClasses[1],legendClasses[0]],fontsize = 20)
            if save_plots:
                plt.savefig(os.path.join(case_folder,feat + '_' + cat+'_manifold.png'),bbox_inches='tight',dpi = 300)
                #plt.savefig(os.path.join(case_folder,'manifold2.png'))
            else:
                plt.show()
        
