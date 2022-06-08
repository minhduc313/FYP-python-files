#ploting parameter

import functools
from math import radians
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os
from collections import Counter
import sys
from scipy.signal import argrelextrema
from matplotlib.colors import LogNorm, Normalize
from multiprocessing import Pool
import time
from functools import partial
from tqdm import tqdm

#2ind
#filename= os.path.dirname(os.path.abspath(__file__))+"/ACDC_abcsmc_2_2022-03-24"

#1ind
filename= os.path.dirname(os.path.abspath(__file__))+"/ACDC_abcsmc_1_2022-03-30"

#n=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
n=['final']
n=n[0]
#sys.path.insert(0, '/users/ibarbier/AC-DC/'+filename)
sys.path.insert(0, 'C:/Users/Administrator/Desktop/Modeling/AC-DC/'+filename)


#Uncomment for 1 ind version

import model_equation_1_ind as meq
parlist = meq.parlist


'''
import model_equation as meq
parlist= meq.parlist
'''

######################################################################33
#########################################################################
###########################################################################

def load(number= n,filename=filename,parlist=parlist):
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
   
        p0=pars_to_dict(p0)
        p.append(p0)

    
    return p, df


def loadBifurcation(number= n,filename=filename):
    index=['0','1000','2000','3000','4000']
    max_stability =np.array([])
    count_bifurcation =np.array([])
    bifurcation_transition =np.array([])
    for i in index:
        path1 = filename+'/bifurcation/' +str(n)+'/'+i +'_'+ number +'max_stability.out'
        path2 = filename+'/bifurcation/' +str(n)+'/'+i +'_'+ number +'count_bifurcation.out'
        path3 = filename+'/bifurcation/' +str(n)+'/'+i +'_'+ number +'bifurcation_transition.out'
        output1= np.loadtxt(path1)
        output2= np.loadtxt(path2)
        output3= np.loadtxt(path3)
        max_stability=np.append(max_stability,output1)
        if len(count_bifurcation)==0:
            count_bifurcation=output2
        else:
            count_bifurcation=np.vstack((count_bifurcation,output2))
        if len(bifurcation_transition)==0:
            bifurcation_transition=output3
        else:
            bifurcation_transition=np.vstack((bifurcation_transition,output3))

    return max_stability,count_bifurcation,bifurcation_transition




def pars_to_dict(pars):
### This function is not necessary, but it makes the code a bit easier to read,
### it transforms an array of pars e.g. p[0],p[1],p[2] into a
### named dictionary e.g. p['k0'],p['B'],p['n'],p['x0']
### so it is easier to follow the parameters in the code
    dict_pars = {}
    for ipar,par in enumerate(parlist):
        dict_pars[par['name']] = pars[ipar] 
    return dict_pars
##plotting part


     


def bifurcation_Xplot(ARA,n,filename,pars,c):
    sizex=round(np.sqrt(len(pars)))
    sizey=round(np.sqrt(len(pars))+0.5)

    for pi,p in enumerate(pars):
        print(pi)
        s,eig,un,st,os,hc,M,m=calculateALL2(ARA,p,dummy=pi) 
        plt.subplot(sizex,sizey,pi+1)
        #plt.tight_layout()
        for i in np.arange(0,un.shape[1]):
            plt.plot(ARA,un[:,i,0],'--',c='orange',linewidth=1)
            plt.plot(ARA,st[:,i,0],'-r',linewidth=1)
            plt.plot(ARA,os[:,i,0],'--b',linewidth=1)
            plt.plot(ARA,hc[:,i,0],'--g',linewidth=1)
            plt.plot(ARA,M[:,i,0],'-b',linewidth=1)
            plt.plot(ARA,m[:,i,0],'-b',linewidth=1)
            plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
        plt.tick_params(axis='both', which='major', labelsize=2)
        plt.yscale("log")
        plt.xscale("log")
    plt.savefig(filename+"/bifurcation/"+'/{}/'.format(n)+c+'_Bifurcation.pdf', bbox_inches='tight')
    plt.close()
   # plt.show()


def bifurcation_Xplot_test(ARA,n,filename,p,c):


        s,eig,un,st,os,hc,M,m=calculateALL2(ARA,p,dummy=0) 
        #plt.tight_layout()
        for i in np.arange(0,un.shape[1]):
            plt.plot(ARA,un[:,i,0],'--o',c='orange',linewidth=2)
            plt.plot(ARA,st[:,i,0],'-or',linewidth=2)
            plt.plot(ARA,os[:,i,0],'--ob',linewidth=2)
            plt.plot(ARA,hc[:,i,0],'--og',linewidth=2)
            plt.plot(ARA,M[:,i,0],'-ob',linewidth=2)
            plt.plot(ARA,m[:,i,0],'-ob',linewidth=2)
            plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
        plt.tick_params(axis='both', which='major')
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("ARA")
        plt.savefig(filename+"/bifurcation/"+c+'_Bifurcation.png', bbox_inches='tight')
        plt.close()
   #     plt.show()


def par_plot(df,name,nb,parlist,namelist):
    #plt.plot(df['K_ARAX'],df['K_ARAY'],'ro')
    fonts=2
 
    for i,par1 in enumerate(namelist):
        for j,par2 in enumerate(namelist):
            plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
            if i == j :
                plt.hist(df[par1])
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
            else:
                plt.scatter(df[par1],df[par2], c=df['dist'], s=0.001, cmap='viridis')# vmin=mindist, vmax=maxdist)
                plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
                plt.ylim((parlist[j]['lower_limit'],parlist[j]['upper_limit']))
            if i > 0 and j < len(namelist)-1 :
                plt.xticks([])
                plt.yticks([])
            else:
                if i==0 and j!=len(namelist)-1:
                    plt.xticks([])
                    plt.ylabel(par2,fontsize=fonts)
                    plt.yticks(fontsize=fonts,rotation=90)
                if j==len(namelist)-1 and i != 0:
                    plt.yticks([])
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                else:
                    plt.ylabel(par2,fontsize=fonts)
                    plt.xlabel(par1,fontsize=fonts)
                    plt.xticks(fontsize=fonts)
                    plt.yticks(fontsize=4,rotation=90)                 
    plt.savefig(name+"/bifurcation/"+nb+'_par_plot.pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

##############################################3Bifurcation part 
################################################################################

def getminmax(X,Y,Z,transient):
    M=np.ones(3)*np.nan
    m=np.ones(3)*np.nan
    M[0]=max(X[transient:])
    M[1]=max(Y[transient:])
    M[2]=max(Z[transient:])
    m[0]=min(X[transient:])
    m[1]=min(Y[transient:])
    m[2]=min(Z[transient:])

    return M,m

def getpeaks(X,transient):
    max_list=argrelextrema(X[transient:], np.greater)
    maxValues=X[transient:][max_list]
    min_list=argrelextrema(X[transient:], np.less)
    minValues=X[transient:][min_list]

    return maxValues, minValues

def reachss(ssa,X,par,a):
    thr= 0.001
    out=False

    for ss in ssa:
        if np.all(np.isnan(ss)) == False:
            A=meq.jacobianMatrix(a,ss[0],ss[1],ss[2],par)
            eigvals, eigvecs =np.linalg.eig(A)
            sse=eigvals.real
            if np.all(sse<0):
                if abs((X[-2]-ss[0])/X[-2]) < thr:
                         out= True

    return out


def limitcycle(ai,ss,ARA,init,par,dummy,X=[],Y=[],Z=[],transient=500,count=0):
    threshold=0.01
    tt=200
    c=count
    #init=[init[0] + 10e-5,init[1] + 10e-5,init[2] + 10e-5]
    ssa=ss[ai]
    x,y,z=meq.model([ARA[ai]],par,totaltime=tt,init=init)
    X=np.append(X,x)
    Y=np.append(Y,y)
    Z=np.append(Z,z)

    M = m = np.nan

    maxValues, minValues = getpeaks(X,transient)

    if len(minValues)>4 and len(maxValues)>4:
        maximaStability = abs((maxValues[-2]-minValues[-2])-(maxValues[-3]-minValues[-3]))/(maxValues[-3]-minValues[-3]) #didn't take -1 and -2 because, i feel like -1 is buggy sometimes...
        if maximaStability > threshold:
            if reachss(ssa,X,par,ARA[ai])==False:
                #if we didn't reach the stability repeat the same for another 100 time until we reach it
                initt=[X[-2],Y[-2],Z[-2]] #take the -2 instead of -1 because sometimes the -1 is 0 because of some badly scripted part somewhere
                c=c+1
                if c<10:
               # if reachsteadystate(a,initt,par) == False:
                    M,m = limitcycle(ai,ss,ARA,initt,par,dummy,X,Y,Z,count=c)            
                if c==10:
                        #here the issue comes probably from 1) strange peak 2)very long oscillation
                        #here I try to get rid of strange peak , with multiple maxima and minima by peak. for this I take the local maximun and min of each..
                        #the issue here is to create artefact bc in the condition doesn't specify this kind of behaviour

                        maxValues2 = getpeaks(maxValues,0)[0]  
                        minValues2 = getpeaks(minValues,0)[1]
                        if len(minValues2)>4 and len(maxValues2)>4:
                          maximaStability2 = abs((maxValues2[-2]-minValues2[-2])-(maxValues2[-3]-minValues2[-3]))/(maxValues2[-3]-minValues2[-3]) #didn't take -1 and -2 because, i feel like -1 is buggy sometimes...
                          if maximaStability2 < threshold:
                              M,m = getminmax(X,Y,Z,transient=transient)
                          else:
                              #very long oscillation?
                              print("no limit cycle: probably encounter stable point at {} arabinose at p{}".format(ARA[ai],dummy))
                              '''
                              plt.plot(X[transient:])
                              plt.yscale("log")
                              plt.show() 
                              '''
                        else:
                              print("too long oscillation?? at {} arabinose at p{}".format(ARA[ai],dummy))
        else:

            M,m = getminmax(X,Y,Z,transient=transient)
         
    else:
       # print("no enough oscillation: " + str(len(minValues)))
        if reachss(ssa,X,par,ARA[ai])==False:
            #print("homoclinic")
            initt=[X[-2],Y[-2],Z[-2]]
            c=c+1
            if c<10:
                M,m = limitcycle(ai,ss,ARA,initt,par,dummy,X,Y,Z,count=c)  
            if c==10: 
                #very long oscillation?          
                print("error in limit cycle ara = {}, p{}".format(ARA[ai],dummy))
                '''
                plt.plot(X[transient:])
                plt.yscale("log")
                plt.show()
                '''


    return M,m

def getEigen(ARA,par,s):
    A=meq.jacobianMatrix(ARA,s[0],s[1],s[2],par)
    eigvals, eigvecs =np.linalg.eig(A)
    sse=eigvals.real
    return sse #, np.trace(A), np.linalg.det(A)

def getpar(i,df):
    return pars_to_dict(df.iloc[i].tolist())

def calculateALL2(ARA,parUsed, dummy):
    #sort ss according to their stabilitz
    #create stability list of shape : arabinose x steady x X,Y,Z 
    nNode=3 # number of nodes : X,Y,Z
    nStstate= 5 # number of steady state accepted by. to create the storage array
  #  ss=np.ones((len(ARA),nStstate,nNode))*np.nan 
    eig= np.ones((len(ARA),nStstate,nNode))*np.nan 
    unstable=np.ones((len(ARA),nStstate,nNode))*np.nan
    stable=np.ones((len(ARA),nStstate,nNode))*np.nan
    oscillation=np.ones((len(ARA),nStstate,nNode))*np.nan
    homoclincic=np.ones((len(ARA),nStstate,nNode))*np.nan
    M=np.ones((len(ARA),nStstate,nNode))*np.nan
    m=np.ones((len(ARA),nStstate,nNode))*np.nan


    delta=10e-10 #perturbation from ss
    ss=meq.findss2(ARA,parUsed) 
    A=meq.jacobianMatrix2(ARA,ss,parUsed)

    for i in np.arange(0,len(ARA)):
        for j in np.arange(0,ss.shape[1]):
            if np.any(np.isnan(A[i][j]))==False:
                eigvals, eigvecs =np.linalg.eig(A[i][j])
                eig[i,j]=eigvals.real

                if any(eig[i,j]>0):
                    pos=eig[i,j][eig[i,j]>0]
                    if len(pos)==2:
                            if pos[0]-pos[1] == 0:                                
                                init=[ss[i,j,0]+delta,ss[i,j,1]+delta,ss[i,j,2]+delta]
                                M[i,j],m[i,j] = limitcycle(i,ss,ARA,init,parUsed,dummy)###
                                if np.isnan(M[i,j][0]):
                                    homoclincic[i][j]=ss[i,j] 

                                else:
                                    oscillation[i][j]=ss[i,j]
                            else:
                                unstable[i,j]=ss[i,j]
                    else:
                        unstable[i,j]=ss[i,j]
                else:
                    if np.all(eig[i,j]<0):
                        stable[i,j]=ss[i,j]
                    else:
                       unstable[i,j]=ss[i,j]
    return ss,eig,unstable,stable,oscillation,homoclincic,M,m


def findbifurcation(ARA,st,un,os,hc, dummy=0):
    bifu=np.zeros((len(ARA),un.shape[1])) 
    transition=np.zeros((len(ARA))) 
    os2=binarize(os)
    un2=binarize(un)
    st2=binarize(st)
    hc2=binarize(hc)

    #saddle node (1) : when two stable states (st,st) cohexist , transition type 1 st->st
    multistable=np.sum(st2,axis=1)
    multistable[multistable<2]=0
    v1=un2+multistable[:,None]
    v1[v1<3]=0
    v1=st2 +multistable[:,None] + os2
    saddlei_neg=np.where((v1[:-1]-v1[1:])==-3)
    saddlei_pos=np.where((v1[:-1]-v1[1:])==3)
    bifu[saddlei_neg[0]+1,saddlei_neg[1]]=1
    bifu[saddlei_pos[0],saddlei_pos[1]]=1
    transition[saddlei_neg[0]+1] = 1
    transition[saddlei_pos[0]] = 1

    #saddle node (1) : when two stable states (st,os) cohexist , transition type 2 st->os
    v2=np.sum(st2,axis=1)
    v2[v2>1]=1
    v1=os2+v2[:,None]
    v1[v1<2]=0
    saddlei_neg=np.where((v1[:-1]-v1[1:])==-2)
    saddlei_pos=np.where((v1[:-1]-v1[1:])==2)
    bifu[saddlei_neg[0]+1,saddlei_neg[1]]=1
    bifu[saddlei_pos]=1
    transition[saddlei_neg[0]+1] = 2
    transition[saddlei_pos[0]] = 2

    #homclinic (4) : when hc and st cohexist, transition type 4 os->st 
    v1=os2*10+hc2*4+st2
    hci_neg=np.where((v1[:-1]-v1[1:])==-6)
    hci_pos=np.where((v1[:-1]-v1[1:])==6)
    bifu[hci_neg[0]+1,hci_neg[1]]=4
    bifu[hci_pos]=4
    transition[hci_neg[0]+1] = 4
    transition[hci_pos[0]] = 4

    #when no other on same line mean that we have probably homoclinic..
    hci_neg=np.where((v1[:-1]-v1[1:])==-10)
    hci_pos=np.where((v1[:-1]-v1[1:])==10)
    if len(hci_neg[0])>0:
        print("please, check homoclinic for p" + str(dummy))
        bifu[hci_neg[0]+1,hci_neg[1]]=4
        bifu[hci_pos]=4
        transition[hci_neg[0]+1] = 4
        transition[hci_pos[0]] = 4

    #hopf (3) : when hc and st cohexist, transition type 3 st->os withotu hysteresis 
    hfi_neg=np.where((v1[:-1]-v1[1:])==-9)
    hfi_pos=np.where((v1[:-1]-v1[1:])==9)
    bifu[hfi_neg[0]+1,hfi_neg[1]]=3
    bifu[hfi_pos[0],hfi_pos[1]]=3
    transition[hfi_neg[0]+1] = 3
    transition[hfi_pos[0]] = 3

    #print(bifu)
    return bifu, transition

def countbifurcation(bifu):
    saddle = len(bifu[bifu==1])
    Hopf = len(bifu[bifu==3])
    homoclinic = len(bifu[bifu==4])
    return [saddle,Hopf,homoclinic]

    
def getmaxmultistability(os,st):
    v1=os
    v2=v1[:,:,0]/v1[:,:,0]
    v2[np.isnan(v2)]=0
    w1=st
    w2=w1[:,:,0]/w1[:,:,0]
    w2[np.isnan(w2)]=0

    total=np.sum((v2+w2),axis=1)
    return max(total)

def binarize(v1):
    v2=v1[:,:,0]/v1[:,:,0]
    v2[np.isnan(v2)]=0
    return v2


####################################################
###################################################PARALLELISATION HERE
######################################################


def runBifurcation(ARA,filename,n,i):

    p, pdf= load(n,filename,meq.parlist)
    pars = p[(1000*i):(1000+1000*i)]


    sizex = round(np.sqrt(len(pars)))
    sizey = round(np.sqrt(len(pars))+0.5)

    max_stability=[]
    count_bifurcation=[]
    bifurcation_transition=[]

    

    index = i*1000



    for pi,p in enumerate(pars):
        s,eig,un,st,os,hc,M,m=calculateALL2(ARA,p,dummy=pi+index) 
        bifu,trans=findbifurcation(ARA,st,un,os,hc,dummy=pi+index)
        cbifu=countbifurcation(bifu)
        maxst=getmaxmultistability(os,st)
        max_stability.append(maxst)
        count_bifurcation.append(cbifu)
        bifurcation_transition.append(trans)


        plt.subplot(sizex,sizey,pi+1)
        #plt.tight_layout()
        for i in np.arange(0,un.shape[1]):
            plt.plot(ARA,un[:,i,0],'--',c='orange',linewidth=.1)
            plt.plot(ARA,st[:,i,0],'-r',linewidth=.1)
            plt.plot(ARA,os[:,i,0],'--b',linewidth=.1)
            plt.plot(ARA,hc[:,i,0],'--g',linewidth=.1)
            plt.plot(ARA,M[:,i,0],'-b',linewidth=.1)
            plt.plot(ARA,m[:,i,0],'-b',linewidth=.1)
            plt.fill_between(ARA,M[:,i,0],m[:,i,0],alpha=0.2,facecolor='blue')
        plt.text(ARA[0],1,('p'+str(pi+index)),fontsize=1)
        plt.text(ARA[0],10,('S:{} Hf: {} Hmc: {}'.format(cbifu[0],cbifu[1],cbifu[2])),fontsize=1)
        plt.tick_params(axis='both', which='major', labelsize=2)
        plt.yscale("log")
        plt.xscale("log")
    
    np.savetxt(filename+"/bifurcation/"+'/{}/'.format(n)+str(index)+'_'+str(n)+'max_stability.out', max_stability)
    np.savetxt(filename+"/bifurcation/"+'/{}/'.format(n)+str(index)+'_'+str(n)+'count_bifurcation.out', count_bifurcation)
    np.savetxt(filename+"/bifurcation/"+'/{}/'.format(n)+str(index)+'_'+str(n)+'bifurcation_transition.out', bifurcation_transition)


    plt.savefig(filename+"/bifurcation/"+'/{}/'.format(n)+str(index)+'_'+str(n)+'XBifurcationplot.pdf', bbox_inches='tight')
    #plt.show()
    plt.close()
    
    return max_stability, count_bifurcation, bifurcation_transition


def runBifurcations(n,filename,ARAlen=20,ncpus=8):
   # ARA=np.logspace(-4.5,-2.,ARAlen,base=10)
    ARA=np.logspace(-8.,-2.,ARAlen,base=10)
    #p, pdf= load(n,filename,meq.parlist)
    max_stability=[]
    count_bifurcation=[]
    bifurcation_transition=[]
    if os.path.isdir(filename+'/bifurcation'+'/'+str(n)) is False:
        os.mkdir(filename+'/bifurcation/'+'/'+str(n)) #create directory for each generation

    '''
    for i in [0,1,2,3,4]: #how to parallel here?
        psubset=p[(1000*i):(1000+1000*i)]
      
        maxst, cbifu, trans = runBifurcation(ARA,psubset,filename,n,1000*i)
    '''
    i_list = [0,1,2,3,4]

    partial_func = functools.partial(runBifurcation,ARA,filename,n)

    with Pool(processes=ncpus) as pool:
        results = pool.map(partial_func,i_list)

 #       max_stability.append(maxst)
  #      count_bifurcation.append(cbifu)
   #     bifurcation_transition.append(trans)
    
  #  np.savetxt(filename+"/"+'ALL_'+str(n)+'max_stability.out', max_stability,fmt='%s')
  #  np.savetxt(filename+"/"+'ALL_'+str(n)+'count_bifurcation.out', count_bifurcation,fmt='%s')
  #  np.savetxt(filename+"/"+'ALL_'+str(n)+'bifurcation_transition.out', bifurcation_transition,fmt='%s')


def bifplot_parplot_sub(p,pdf,index,filename,n,figname):
    p=np.array(p)
    pars=p[index]
    df=pdf #all data for that generation
    df2=pdf.iloc[index] #data with the specified index
    
    bifurcation_Xplot(ARA,n,filename,pars,c=figname)        
    
    namelist=[]
    for i,par in enumerate(meq.parlist):
           namelist.append(parlist[i]['name'])
    namelist=np.array(namelist)        
    
    fonts=15
    fig = plt.figure(figsize=(30,25))

    for i,par1 in enumerate(namelist):
            for j,par2 in enumerate(namelist):
                plt.subplot(len(namelist),len(namelist), i+j*len(namelist)+1)
                if i == j :
                    #plt.hist(df[par1])
    
                    sns.kdeplot(df[par1],bw_adjust=.8,label=1,linewidth=0.3,color='red',shade=True)
                    sns.kdeplot(df2[par1],bw_adjust=.8,label=2,linewidth=0.3,color='blue',shade=True)
                    plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
                    plt.xticks([])
                    plt.yticks([])
                    plt.ylabel('')
                    plt.xlabel('')
                else:
                    plt.scatter(df[par1],df[par2], c='black', s=1,alpha=0.6)# vmin=mindist, vmax=maxdist)
                    plt.scatter(df2[par1],df2[par2], c='green', s=1,alpha=1)# vmin=mindist, vmax=maxdist)
    
                    plt.xlim((parlist[i]['lower_limit'],parlist[i]['upper_limit']))
                    plt.ylim((parlist[j]['lower_limit'],parlist[j]['upper_limit']))
                if i > 0 and j < len(namelist)-1 :
                    plt.xticks([])
                    plt.yticks([])
                else:
                    if i==0 and j!=len(namelist)-1:
                        plt.xticks([])
                        plt.ylabel(par2,fontsize=fonts)
                        plt.yticks(fontsize=fonts,rotation=90)
                    if j==len(namelist)-1 and i != 0:
                        plt.yticks([])
                        plt.xlabel(par1,fontsize=fonts)
                        plt.xticks(fontsize=fonts)
                    if i==0 and j==len(namelist)-1:
                        plt.ylabel(par2,fontsize=fonts)
                        plt.xlabel(par1,fontsize=fonts)
                        plt.xticks(fontsize=fonts)
                        plt.yticks(fontsize=4,rotation=90)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)                 
    #plt.savefig(filename+"/bifurcation/"+figname+'_par_plot.pdf', bbox_inches='tight')
    plt.savefig(filename+"/bifurcation/"+'/{}/'.format(n)+figname+'_par_plot.png', bbox_inches='tight',dpi=300,orientation='landscape', papertype='a3')

    plt.close()

def fulldf(n,filename):
    p, pdf= load(n,filename,meq.parlist)
    maxst,cbifu,bifutr=loadBifurcation(n,filename)
    pdf=pdf.sort_values('dist',ascending=True) #to be in same order than p
    pdf['max_stability']=maxst
    pdf['saddle']=cbifu[:,0]
    pdf['hopf']=cbifu[:,1]
    pdf['homoclinic']=cbifu[:,2]
    return pdf,bifutr



def ACDC_select(bifutr):
    ACDC_onlyHopf_index=[]
    ACDC_index=[]

    hopf_index=np.where(bifutr==3) #index of where there is hopf
    for i in np.arange(0,len(hopf_index[0])):
        #divide after and before 1st hopf
        sub_after=bifutr[hopf_index[0][i],hopf_index[1][i]+1:]
        sub_before=bifutr[hopf_index[0][i],:hopf_index[1][i]]

        #there is only one transition from stable to oscillation
        if len(np.where(sub_after==2)[0])==1:

            #there is a least 2 point of bistability
            if np.any(np.where(sub_after>2)[0] - np.where(sub_after==2)[0] >2):

                #if oscillation finish by hopf, it should have only one saddle node after
                i2= np.where(sub_after==3)[0]
                if len(i2)>0 :
                    if np.sum(sub_after[i2[0]+1:]) == 1 :    
                        ACDC_index.append(hopf_index[0][i])
                        if np.sum(sub_before) == 0 :
                            ACDC_onlyHopf_index.append(hopf_index[0][i])
                #if finish by homoclinic
                else :
                    ACDC_index.append(hopf_index[0][i])
                    if np.sum(sub_before) == 0 :
                            ACDC_onlyHopf_index.append(hopf_index[0][i])
    return   ACDC_onlyHopf_index, ACDC_index









################################BUILDING AREA
if os.path.isdir(filename+'/bifurcation') is False: ## if 'smc' folder does not exist:
  os.mkdir(filename+'/bifurcation') ## create it, the output will go there

n_list=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

#n=n[0]
ARAlen=50
#ARAlen=10

#ARA=np.logspace(-4.5,-2.,ARAlen,base=10)
ARA=np.logspace(-8.,-2.,ARAlen,base=10)

'''
for n in tqdm(n_list):
    runBifurcations(n,filename,ARAlen=50)
'''

bifudf,bifutr = fulldf(n,filename)
ACDC_onlyHopf_index, ACDC_index = ACDC_select(bifutr)
p,pdf=load(number=n,filename=filename,parlist=parlist)
print(bifudf)
print(pdf)

#bifurcation_Xplot_test(ARA,n,filename,p[0],'test')

#bifplot_parplot_sub(p,pdf,ACDC_index,filename,n,'2acdc')


#print(pdf.iloc[ACDC_index])
#ACDC_pars = [p[i] for i in ACDC_index]
#onlyHopf_par = [p[i] for i in ACDC_onlyHopf_index]

#bifurcation_Xplot(ARA,n,filename,ACDC_pars,'ACDC_only')

#bifurcation_Xplot(ARA,n,filename,onlyHopf_par,'only_Hopf')
#bifplot_parplot_sub(p,pdf,ACDC_onlyHopf_index,filename,n,'2acdclike_onlyhopf')
#bifplot_parplot_sub(p,pdf,ACDC_index,filename,n,'2acdclike')


##############################################################################################################3   







    
    
