#Bayes factor 

from posixpath import relpath
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os
import sys
from matplotlib.colors import LogNorm, Normalize
import multiprocessing
import time
from functools import partial
from tqdm import tqdm
from shapely.geometry import Point, LineString
from scipy.integrate import simps
from numpy import trapz
from scipy.stats import norm, uniform, multivariate_normal


import model_equation_original as meq
import model_equation_1ind_original as meq_1
parlist=meq.parlist
parlist_1ind = meq_1.parlist

filename_2 =   os.path.dirname(os.path.abspath(__file__))+'/ACDC_abcsmc_2_2022-03-24'
filename_1 =   os.path.dirname(os.path.abspath(__file__))+'/ACDC_abcsmc_1_2022-03-30'
def load(number,filename=filename_2,parlist=parlist):
    namelist=[]
    for i,par in enumerate(parlist):
        namelist.append(parlist[i]['name'])
    
    path = filename+'/smc/pars_' + number + '.out'
    dist_path = filename+'/smc/distances_' + number + '.out'

    raw_output= np.loadtxt(path)
    dist_output= np.loadtxt(dist_path)
    df = pd.DataFrame(raw_output, columns = namelist)
    df['dist']=dist_output
    df=df.sort_values('dist',ascending=False)
    distlist= sorted(df['dist'])
    p=[]
    for dist in distlist:
        
        p_0=df[df['dist']==dist]
        p0=[]
        for n in namelist:
          p0.append(p_0[n].tolist()[0])
        p0=pars_to_dict(p0,parlist)
        p.append(p0)

    
    return p, df 

def pars_to_dict(pars,parlist):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars







import warnings
warnings.filterwarnings("ignore")

def posterior_likelihood(target_likelihood=0.9,threshold_step=0.001,filename=filename_2,parlist=parlist,generation='final'):

    #Creating dictionary with parameter names as keys for output later
    par_names=[]
    for ipar,par in enumerate(parlist):
        par_names.append(par['name'])
    credible_intervals_dict = dict.fromkeys(par_names)

    ###

    p, pdf= load(generation,filename,parlist)
    posterior_likelihood.par_size = len(pdf)
    for ipar,par in enumerate(parlist):

    	#Create kde plot and obtain line cooordinates for each parameters
        kde_plot =sns.kdeplot(pdf[par['name']],common_norm=True) 
        kde_line = kde_plot.lines[0]
        x, y = kde_line.get_data() #Coordinates of kdeplot line (length =200)
        #plt.plot(x,y)

        #Horizontal line coordinates
        threshold= 0.01          #initial probability threshold (too small value may not work for certain parameters(e.g:n_zx,n_xz,n_xy))
        likelihood = 1           #initial likelihood

        #probability threshold search step(smaller-> higher accuracy). 
        #0.001 gives final likelihood accuaracy up to 2 decimal places
        step = threshold_step        
        
        while likelihood>target_likelihood:
            #Coordinates for horizontal line 
            x1 = np.linspace(start=par['lower_limit']-1,stop=par['upper_limit']+1,num=200) # +1 and -1 to make sure line intersects with curve fully
            y1 = np.ones(200)*threshold 
            #plt.plot(x1,y1)
            threshold+= step   #update (increasing) probability threshold after each round

            #Find intersections
            line = LineString(np.column_stack((x,y))) #kde curve
            line_1 = LineString(np.column_stack((x1,y1))) #horizontal line
            intersection = line_1.intersection(line)
            first_idx=[intersection[1].xy[0][0],intersection[1].xy[1][0]]
            second_idx=[intersection[0].xy[0][0],intersection[0].xy[1][0]]

            if first_idx[0]>second_idx[0]: #X coordinates of intersections
                high_idx = first_idx[0]
                low_idx  = second_idx[0]
            else:
                high_idx=second_idx[0]
                low_idx=first_idx[0]

            #print('low intersection:',low_idx)
            #print('high intersection',high_idx)

            '''
            #Find area under curve between 2 intersections (not nescessary)

            index = np.where((x<high_idx)&(x>low_idx)) 
            #print('index', index)
            #plt.plot(x[index],y[index])

            dx= x[2]-x[1]  #difference between 2 consecutive x coordinates to calculate area under curve (not nescessary)
            area = trapz(y[index],dx=dx)
            area_2 = simps(y[index], dx=dx)
            #print('area', area,area_2)
            '''

            #Calculate likelihood

            within_par=np.count_nonzero((pdf[par['name']]<high_idx) & (pdf[par['name']]>low_idx))
            #print('within paramters:', within_par) #Number of parameters within range
            likelihood = within_par/len(pdf[par['name']]) #likelihood at given threshold probability 
            #print(likelihood)

        '''
        print('for parameter:', par['name'])
        print('final likelihood ',likelihood)
        print('final threshold',threshold)
        print('low intersection:',low_idx)
        print('high intersection',high_idx)
        '''
        #plt.plot(x1,y1) #threshold probability line at target likelihood
        plt.clf() #clear plot for next parameter iteration

        credible_intervals_dict[par['name']]=[low_idx,high_idx]

    return credible_intervals_dict




def prior_likelihood(par_range_dict,parlist=parlist):
    par_names=[]
    for ipar,par in enumerate(parlist):
        par_names.append(par['name'])
    prior_likelihood_dict = dict.fromkeys(par_names)

    ####
    summary_likelihood = 1 
    for ipar,par in enumerate(parlist):
                
        prior_distribution= uniform.rvs(loc=par['lower_limit'],scale=par['upper_limit']+1 - par['lower_limit']-1,size=posterior_likelihood.par_size )
        #plt.hist(prior_distribution)

        within_par_count = np.count_nonzero((prior_distribution > par_range_dict[par['name']][0]) & (prior_distribution< par_range_dict[par['name']][1]))
        prior_likelihood = within_par_count/len(prior_distribution)
        prior_likelihood_dict[par['name']] = prior_likelihood
        summary_likelihood *= prior_likelihood

    #print('Prior likelihood summary:',summary_likelihood)

    return prior_likelihood_dict, summary_likelihood



###################

'''
bayes_factor = two_ind_prior[1]/one_ind_prior[1]
print(bayes_factor)

print('one_ind_evidence:')
print(one_ind_prior[1])


print('two_ind evidence:')
print(two_ind_prior[1])
'''

os.makedirs(filename_1+'/bayes', exist_ok=True)  
os.makedirs(filename_2+'/bayes',exist_ok=True)

n_list =['18']
for n in n_list:
    #two_ind_posterior= posterior_likelihood(generation=n,filename=filename_2)
    one_ind_posterior= posterior_likelihood(generation=n,filename=filename_1,parlist=parlist_1ind)

    #two_ind_prior = prior_likelihood(two_ind_posterior)
    one_ind_prior = prior_likelihood(one_ind_posterior,parlist=parlist_1ind)





    #print('two_ind credible intervals'+str(two_ind_posterior))
    #print('one_ind credible intervals'+str(one_ind_posterior))


    prior_df1 = pd.DataFrame(data=one_ind_prior[0],index=['Likelihood']).transpose()
    #prior_df2 = pd.DataFrame(data=two_ind_prior[0],index=['Likelihood']).transpose()
    credible_intervals_1 = pd.DataFrame(data=one_ind_posterior,index=['Low limit','High limit']).transpose()
    #credible_intervals_2 = pd.DataFrame(data=two_ind_posterior,index=['Low limit','High limit']).transpose()


    credible_intervals_1.to_csv(filename_1+'/bayes/cred_{}.csv'.format(n))
    #credible_intervals_2.to_csv(filename_2+'/bayes/cred_{}.csv'.format(n))
    prior_df1.to_csv(filename_1+'/bayes/prior_likelihood_{}.csv'.format(n))
    #prior_df2.to_csv(filename_2+'/bayes/prior_likelihood_{}.csv'.format(n))




