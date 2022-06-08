#Importance sampling for Bayes facotr



import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.optimize import minimize
import sys, ast
from random import choices, seed, random
from tqdm import tqdm
from p_tqdm import p_umap, p_map  # parallel code
from functools import partial
import matplotlib.pyplot as plt
import os
import pandas as pd
import multiprocessing
import time
import datetime

today_date = str(datetime.date.today())

version = 'ACDC_abcsmc_2_2022-03-24'
pl = 'final'  # prior_label
ncpuss = 8

# sys.path.insert(0, '/users/ibarbier/AC-DC/'+version+'/')
# sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+version)
import model_equation_original as model_equation

parlist = model_equation.parlist
x_data = model_equation.ARA

initdist = model_equation.initdist
finaldist = 40


def pars_to_dict(pars):
    ### This function is not necessary, but it makes the code a bit easier to read,
    ### it transforms an array of pars e.g. p[0],p[1],p[2] into a
    ### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
    ### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar, par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar]
    return dict_pars


def sampleprior():
    ### Generate a random parameter inside the limits stablished. The shape of the distribution can be changed if required
    prior = []
    for ipar, par in enumerate(parlist):
        prior.append(uniform.rvs(loc=par['lower_limit'],
                                 scale=par['upper_limit'] - par['lower_limit']))
    return prior


def evaluateprior(pars, model=None):
    ### Given a set of parameters return how probable is given the prior knowledge.
    ### If the priors are uniform distributions, this means that it will return 0 when a parameter
    ### outside the prior range is tested.
    prior = 1
    for ipar, par in enumerate(parlist):
        prior *= uniform.pdf(pars[ipar], loc=par['lower_limit'],
                             scale=par['upper_limit'] - par['lower_limit'])
    return prior


def GeneratePar(x_data, iter,
                processcall=0, previousparlist=None, previousweights=None,
                eps_dist=10000, kernel=None):
    # This function generates a valid parameter given a set of sampled pars previousparlist,previousweights
    # that is under a threshold eps_dist

    # processcall is a dummy variable that can be useful when tracking the function performance
    # it also allows the use of p_tqdm mapping that forces the use of an iterator

    # get current process
    # print(multiprocessing.current_process())

    # @EO: to compare the two versions, set the seeds
    # seed(iter) # setting random seeds for each thread/process to avoid having the same random sequence in each thread
    # np.random.seed(iter)
    seed()  # setting random seeds for each thread/process to avoid having the same random sequence in each thread
    np.random.seed()

    evaluated_distances = []
    d = eps_dist + 1  # original distance is beyond eps_dist
    
  # until a suitable parameter(below the threshold) is found

    if previousparlist is None:  # if is the first iteration
        proposed_pars = sampleprior()
    else:
        selected_pars = choices(previousparlist, weights=previousweights)[0]  # select a point from the previous sample
        proposed_pars = selected_pars + kernel.rvs()  # and perturb it

    if (evaluateprior(proposed_pars, model_equation.model) > 0):  
        # here
        p = pars_to_dict(proposed_pars)
        d = model_equation.distance(x_data, p)
        evaluated_distances.append(d)
    


    # Calculate weight
    if previousparlist is None:
        weight = 1
    else:
        sum_denom = 0
        for ipars, pars in enumerate(previousparlist):
            kernel_evaluation = kernel.pdf(proposed_pars - pars)
            sum_denom += kernel_evaluation * previousweights[ipars]

        weight = evaluateprior(proposed_pars) / sum_denom

    return proposed_pars, d, weight, evaluated_distances


def GeneratePars(x_data, ncpus,
                 previousparlist=None, previousweights=None, eps_dist=10000,
                 Npars=1000, kernelfactor=1.0):
    # @EO: to compare the 2 versions, set the seed
    # np.random.seed(0)

    ## Call to function GeneratePar in parallel until Npars points are accepted

    if previousparlist is not None:
        previouscovar = 2.0 * kernelfactor * np.cov(np.array(previousparlist).T)
        # print('covariance matrix previous parset:',previouscovar)
        kernel = multivariate_normal(cov=previouscovar)

    else:
        kernel = None

    trials = 0

    # for N in tqdm(range(Npars)): ## not parallel version
    #   GenerateParstePar(0,model = 'Berg',gamma = gamma, previousparlist = previousparlist,
    #   previousweights = previousweights, eps_dist = eps_dist, kernel = kernel)

    start_time = time.time()
    results = []
    accept=0

    if 1 == 1:
        # @EO: to compare with new implementation use p_map() instead of p_umap()
        # results = p_map(
        results = p_umap(
            partial(GeneratePar, x_data, previousparlist=previousparlist,
                    previousweights=previousweights, eps_dist=eps_dist, kernel=kernel),
            range(Npars), num_cpus=ncpus)
    else:
        pool = multiprocessing.Pool(ncpus)
        # @EO: consider imap or shuffle results
        # @EO: to compare to orginal pool.map with chunksize=1
        results = pool.map(func=partial(GeneratePar, x_data, previousparlist=previousparlist,
                                        previousweights=previousweights, eps_dist=eps_dist, kernel=kernel),
                           iterable=range(Npars), chunksize=10)
        pool.close()
        pool.join()
    end_time = time.time()
    print(f'>>>> Loop processing time: {end_time - start_time:.3f} sec on {ncpus} CPU cores.')
    print('Accepted particles:'+str(accept))
    accepted_distances = [result[1] for result in results]
    evaluated_distances = [res for result in results for res in result[3]]  # flatten list
    newparlist = [result[0] for result in results]
    newweights = [result[2] for result in results]

    newweights /= np.sum(newweights)  # Normalizing
    print("acceptance rate:", Npars / len(evaluated_distances))
    print("min accepted distance: ", np.min(accepted_distances))
    print("median accepted distance: ", np.median(accepted_distances))
    print("median evaluated distance: ", np.median(evaluated_distances))

    return (newparlist, newweights, accepted_distances, Npars / len(evaluated_distances))


def Sequential_ABC(x_data, ncpus,
                   initial_dist=20.0, final_dist=4.0, Npars=1000, prior_label=None,
                   adaptative_kernel=True):
    ## Sequence of acceptance threshold start with initial_dis and keeps on reducing until
    ## a final threshold final_dist is reached.

    ## prior_label can be used to restart a sampling from a previous prior distribution in case
    ## further exploration with a lower epsilon is needed

    distance = initial_dist
    not_converged = True
    last_round = True
    kernelfactor = 1.0

    if prior_label is None:
        pars = None
        weights = None
        idistance = 0
    else:  # a file with the label is used to load the posterior, use always numerical (not 'final')
        '''
        pars = np.loadtxt('smc_'+'/pars_{}_{}_{}_{}.out'.format(model_equation.model,sto,gamma,prior_label))
        weights = np.loadtxt('smc_'+'/weights_{}_{}_{}_{}.out'.format(model_equation.model,sto,gamma,prior_label))
        accepted_distances = np.loadtxt('smc_'+'/distances_{}_{}_{}_{}.out'.format(model_equation.model,sto,gamma,prior_label))
        '''
        pars = np.loadtxt(version + '/smc' + '/pars_{}.out'.format(prior_label))
        weights = np.loadtxt(version + '/smc' + '/weights_{}.out'.format(prior_label))
        accepted_distances = np.loadtxt(version + '/smc' + '/distances_{}.out'.format(prior_label))
        #distance = np.median(accepted_distances)
        idistance = prior_label

    while not_converged:

        #idistance += 1

        # @EO: for testing, limit the number of iterations
        '''
        if idistance == 3:
            last_round = True
        '''

        print("SMC step with target distance: {}".format(distance))
        pars, weights, accepted_distances, acceptance = GeneratePars(x_data, ncpus,
                                                                     previousparlist=pars,
                                                                     previousweights=weights, eps_dist=1.1,   #eps_dist set as default(10000)
                                                                     Npars=Npars, kernelfactor=kernelfactor)
        proposed_dist = np.median(accepted_distances)
        if last_round is True:
            not_converged = False
            label = 'Importance_sampling'
        else:
            label = idistance
        if proposed_dist < final_dist:
            distance = final_dist
            last_round = True
        else:
            distance = proposed_dist
        np.savetxt(version + '/smc' + '/pars_{}.out'.format(label), pars)
        np.savetxt(version + '/smc' + '/weights_{}.out'.format(label), weights)
        np.savetxt(version + '/smc' + '/distances_{}.out'.format(label), accepted_distances)

        if acceptance < 0.1 and kernelfactor > 0.1 and adaptative_kernel:  # change here to have lower condition
            kernelfactor = kernelfactor * 0.7
            print('Reducing kernel width to : ', kernelfactor)
        elif acceptance > 0.5 and kernelfactor < 1 and adaptative_kernel:
            kernelfactor = kernelfactor / 0.7
            print('Increasing kernel width to : ', kernelfactor)

################

def main(argv):
    if os.path.isdir(version) is False:  ## if 'smc' folder does not exist:
        os.mkdir(version)  ## create it, the output will go there

    if os.path.isdir(version + '/smc') is False:  ## if 'smc' folder does not exist:
        os.mkdir(version + '/smc')  ## create it, the output will go there

    Sequential_ABC(x_data, ncpus=ncpuss, initial_dist=initdist, final_dist=finaldist, prior_label=pl, Npars=50000, #number of sampling time#Adjusted to 5000 to increase exploration
                   adaptative_kernel=False)
    # adaptative_kernel allows to decrease the exploration space according to the probability to find a new parameter. However I find it too much strigent.


if __name__ == "__main__":
    main(sys.argv[1:])
