
## equation for X only 1ind

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
import numpy as np
from scipy import optimize
from scipy.optimize import brentq


#par for abc_smc
initdist=40.
finaldist=1.0
priot_label=None
dtt=0.1
tt=120 #totaltime
tr=20 #transient time
node="X"

#list for ACDC
parlist = [ 
    #first node X param
    {'name' : 'K_ARAX', 'lower_limit':-4.0,'upper_limit':-1.0}, 
    {'name' : 'n_ARAX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XY','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_XY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_XZ','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_XZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta/alpha_X','lower_limit':0.0,'upper_limit':4.0},

    #Seconde node Y param
#    {'name' : 'K_ARAY', 'lower_limit':-4.0,'upper_limit':-1.0}, 
#    {'name' : 'n_ARAY','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'K_YZ','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_YZ','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta/alpha_Y','lower_limit':0.0,'upper_limit':4.0},

    #third node Z param
    {'name' : 'K_ZX','lower_limit':-10.0,'upper_limit':2.0},
    {'name' : 'n_ZX','lower_limit':0.5,'upper_limit':2.0},
    {'name' : 'beta/alpha_Z','lower_limit':0.0,'upper_limit':4.0},
]


ARA=np.logspace(-4.5,-2.,10,base=10) 


def Flow(X,Y,Z,ARA,par):
    #simplify version with less parameter    
    flow_x= 1 + (10**par['beta/alpha_X']-1)*(np.power(ARA,par['n_ARAX'])/( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))) 
    flow_x = flow_x / ( 1 + np.power((Z/10**(par['K_ZX'])),par['n_ZX']))
    flow_x = flow_x - X

    flow_y = 10**par['beta/alpha_Y']
    flow_y = flow_y / ( 1 + np.power(X/10**(par['K_XY']),par['n_XY']))
    flow_y = flow_y - Y

    flow_z = 10**par['beta/alpha_Z']/( 1 + np.power(Y/10**par['K_YZ'],par['n_YZ']))
    flow_z = flow_z /( 1 + np.power(X/10**par['K_XZ'],par['n_XZ']))
    flow_z = flow_z - Z

    return flow_x,flow_y,flow_z


def Integration(Xi,Yi,Zi, totaltime, dt, ch , pars ):
    X=Xi
    Y=Yi
    Z=Zi
    ti=0
    while (ti<totaltime):
        flow_x,flow_y,flow_z = Flow(Xi,Yi,Zi,ch,pars)
        Xi = Xi + flow_x*dt
        Yi = Yi + flow_y*dt
        Zi = Zi + flow_z*dt
        
        X=np.vstack((X,Xi))
        Y=np.vstack((Y,Yi))
        Z=np.vstack((Z,Zi))

        ti=ti+dt

    return X,Y,Z


def distance(x,pars,totaltime=tt, dt=dtt,trr=tr,N=node):

    X,Y,Z = model(x,pars,totaltime,dt)
    
    # transient time / dt
    transient = int(trr/dt)
 
    #range where oscillation is expected
    oscillation_ara=[2,7]
    
    A=X #default
    if N== "X":
      A=X
    if N== "Y":
      A=Y
    if N== "Z":
      A=Z    
    d_final=0
        
    for i in range(0,len(x)):
             # for local maxima
             max_list=argrelextrema(A[transient:,i], np.greater)
             maxValues=A[transient:,i][max_list]
             # for local minima
             min_list=argrelextrema(A[transient:,i], np.less)
             minValues=A[transient:,i][min_list]
     
     
             if i>oscillation_ara[0] and i<oscillation_ara[1]:           
                
                 if len(maxValues)>0 and len(maxValues)<2 and len(minValues)<2:
                     d= 1/len(maxValues) + 1
                
                 if len(maxValues)>=3 and len(minValues)>=3:  #if there is more than one peak
                    # print("max: " + str(len(maxValues)) + "   min:" + str(len(minValues)))
     
                     #here the distance is only calculated on the last two peaks
                     #d2=abs((maxValues[-1]-minValues[-1]) - (maxValues[-2]-minValues[-2]))/(maxValues[-2]-minValues[-2])  #maybe issue still here ? 
                     #d3=2*(minValues[-1])/(minValues[-1]+maxValues[-1]) #Amplitude of oscillation
                     
                     d2=abs((maxValues[-2]-minValues[-2]) - (maxValues[-3]-minValues[-3]))/(maxValues[-2]-minValues[-2])  #maybe issue still here ? 
                     d3=2*(minValues[-2])/(minValues[-2]+maxValues[-2]) #Amplitude of oscillation
                     d= d2+d3
     
                 else:
                     d=10 # excluded all the one without oscillation 
                     #d=abs(max(X[transient:,i])-max(X[transient:,(i+1)]))/max(X[transient:,i])
                     #this number can be tuned to help the algorythm to find good parameter....
                 #d=0 #v22 DC only
                 
             if i<oscillation_ara[0] or i>oscillation_ara[1]:  #notice than 2 inducer concentration are not precised here. no leave some place at transition dynamics
                 d1=  len(minValues)/(1+len(minValues)) # v14,21 with len(minValues)/(1+len(minValues)) #v15 10*len(minValues)/(1+len(minValues))
                 d2=  2*(max(A[transient:,i])-min(A[transient:,i]))/(max(A[transient:,i])+min(A[transient:,i]))
                 d= d1+d2
                 #d= 0 #v20 try to have repressilator
                
     
                 
             if i==oscillation_ara[0] or i==oscillation_ara[1]: 
                 d=0
            # print(d)
             d_final=d_final+d
        
    
   # d= 10*A[-1,0]/(A[-1,-1]+A[-1,0]) #try to valorise increase behaviour compare to dead one
   # print("diff   ", d)
    d_final=d_final+d
    
    if N=="ALL":
      dy=distance(x,pars,totaltime=tt, dt=dtt,trr=tr,N="Y")
      dz=distance(x,pars,totaltime=tt, dt=dtt,trr=tr,N="Z")
      d_final=d_final+dy+dz
        
    return d_final


def model(x,pars,totaltime=tt, dt=dtt,init=[0.2,0,0]):
    Xi=np.ones(len(x))*init[0]
    Yi=np.ones(len(x))*init[1]
    Zi=np.ones(len(x))*init[2]
    X,Y,Z = Integration(Xi,Yi,Zi,totaltime,dt,x,pars)
    return X,Y,Z


def stability(ARA,par,ss=0):
    if ss==0:
        ss= findss(ARA,par)
    eigens=[]
    for i,s in enumerate(ss): 
        A=jacobianMatrix(ARA,s[0],s[1],s[2],par)
        eigvals, eigvecs =np.linalg.eig(A)
        sse=eigvals.real
       # print(sse)
        eigens.append(sse)
        
    return eigens #, np.trace(A), np.linalg.det(A)

def solvedfunction2(Zi,ARA,par):
    #rewrite the system equation to have only one unknow and to be call with scipy.optimze.brentq
    #the output give a function where when the line reach 0 are a steady states
  #  X=np.ones((len(ARA),len(Zi)))
  #  Y=np.ones((len(ARA),len(Zi)))
   # Z=np.ones((len(ARA),len(Zi)))
    Zi=np.array([Zi])

    X= 1 + (10**par['beta/alpha_X']-1)*(np.power(ARA,par['n_ARAX'])/( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))) 
    X = X[:, None] / ( 1 + np.power((Zi/10**(par['K_ZX'])),par['n_ZX'])) 

    Y = 10**par['beta/alpha_Y']
  #  Y = np.array(Y)
    Y = Y / ( 1 + np.power(X/10**(par['K_XY']),par['n_XY']))

    Z = 10**par['beta/alpha_Z']/( 1 + np.power(Y/10**(par['K_YZ']),par['n_YZ']))
    Z = Z /( 1 + np.power(X/10**(par['K_XZ']),par['n_XZ']))
    func = Zi - Z
    return func 

def solvedfunction(Zi,ARA,par):
    Zi=np.array([Zi])

    X= 1 + (10**par['beta/alpha_X']-1)*(np.power(ARA,par['n_ARAX'])/( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))) 
    X = X / ( 1 + np.power((Zi/10**(par['K_ZX'])),par['n_ZX'])) 

    Y = 10**par['beta/alpha_Y']
    Y = Y / ( 1 + np.power(X/10**(par['K_XY']),par['n_XY']))

    Z = 10**par['beta/alpha_Z']/( 1 + np.power(Y/10**(par['K_YZ']),par['n_YZ']))
    Z = Z /( 1 + np.power(X/10**(par['K_XZ']),par['n_XZ']))
    func = Zi - Z
    return func    

def findss2(ARA,par):   
    #function to find steady state
    #1. find where line reached 0
    Zi=np.logspace(-14,5,500,base=10)
    f=solvedfunction2(Zi,ARA,par)
    x=f[:,1:]*f[:,0:-1] #when the output give <0, where is a change in sign, meaning 0 is crossed
    index=np.where(x<0)   
    nNode=3 # number of nodes : X,Y,Z
    nStstate= 5 # number of steady state accepted by. to create the storage array
    ss=np.ones((len(ARA),nStstate,nNode))*np.nan  

    for i,ai in enumerate(index[0]):
        Z=brentq(solvedfunction, Zi[index[1][i]], Zi[index[1][i]+1],args=(ARA[ai],par)) #find the value at 0
        #now we have the other ss
        X= 1 + (10**par['beta/alpha_X']-1)*(np.power(ARA[ai],par['n_ARAX'])/( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA[ai],par['n_ARAX']))) 
        X = X / ( 1 + np.power((Z/10**(par['K_ZX'])),par['n_ZX']))
        Y = 10**par['beta/alpha_Y']
        Y = Y / ( 1 + np.power(X/10**(par['K_XY']),par['n_XY']))
        ss= addSSinGoodOrder(ss,np.array([X,Y,Z]),ai,0,beforeXvalues=[])

    return ss
    
def addSSinGoodOrder(vector,values,ai,pos,beforeXvalues=[]):

    if ai ==0:
        if np.isnan(vector[ai,pos,0]):
            vector[ai,pos]=values
        else:
            pos=pos+1
            vector=addSSinGoodOrder(vector,values,ai,pos)
    else:
        if len(beforeXvalues) == 0:
            beforeXvalues = vector[ai-1,:,0].copy()
            #print(beforeXvalues)
        if np.all(np.isnan(beforeXvalues)):
            if np.isnan(vector[ai,pos,0]):
                vector[ai,pos]=values
            else:
                 pos=pos+1
                 vector=addSSinGoodOrder(vector,values,ai,pos,beforeXvalues)
        else:
            v1=abs(values[0] - beforeXvalues)
            pos=np.nanargmin(v1)
            if np.isnan(vector[ai,pos,0]):
                vector[ai,pos]=values
            else:
                previousvalues= vector[ai,pos].copy()
               
                v2=abs(previousvalues[0] - beforeXvalues)
                if np.nanmin(v1)<np.nanmin(v2): #if new values closest, takes the place and replace the previous one
                    vector[ai,pos]=values
                    vector=addSSinGoodOrder(vector,previousvalues,ai,pos)
                else:
                    adjusted_beforeXvalues= beforeXvalues.copy()
                    adjusted_beforeXvalues[pos]= np.nan  # remove placed element
                    vector=addSSinGoodOrder(vector,values,ai,pos,adjusted_beforeXvalues)
    return vector 

def jacobianMatrix(ARA,X,Y,Z,par):
##need to update it
    dxdx = -1
    dxdy = 0
    dxdz=-(((np.power(ARA,par['n_ARAX'])*(10**par['beta/alpha_X']-1))/ ( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))+1)*par['n_ZX']*np.power((Z/10**(par['K_ZX'])),par['n_ZX']))
    dxdz=dxdz/(Z*np.power((np.power((Z/10**(par['K_ZX'])),par['n_ZX'])+1),2))

    #dydx=-(((np.power(ARA,par['n_ARAY'])*(10**par['beta/alpha_Y']-1))/ ( np.power(10**par['K_ARAY'],par['n_ARAY']) + np.power(ARA,par['n_ARAY']))+1)*par['n_XY']*np.power((X/10**(par['K_XY'])),par['n_XY']))
    #dydx=dydx/(X*np.power((np.power((X/10**par['K_XY']),par['n_XY'])+1),2)) 

    dydx= 10**par['beta/alpha_Y']*par['n_XY']*np.power((X/10**par['K_XY']),par['n_XY'])
    dydx=dydx/(X*np.power((np.power((X/10**par['K_XY']),par['n_XY'])+1),2))

    dydy=-1
    dydz= 0

    dzdx = -(10**par['beta/alpha_Z']*par['n_XZ']*np.power((X/10**par['K_XZ']),par['n_XZ']))
    dzdx= dzdx /((np.power((Y/10**par['K_YZ']),par['n_YZ'])+1)*X*np.power((np.power(X/10**par['K_XZ'],par['n_XZ'])+1) ,2))

    dzdy= -(10**par['beta/alpha_Z']*par['n_YZ']*np.power((Y/10**par['K_YZ']),par['n_YZ']))
    dzdy= dzdy /((np.power((X/10**par['K_XZ']),par['n_XZ'])+1)*Y*np.power((np.power(Y/10**par['K_YZ'],par['n_YZ'])+1) ,2))
    dzdz = -1
     
    A=np.array(([dxdx,dxdy,dxdz],[dydx,dydy,dydz],[dzdx,dzdy,dzdz]))

    return A

def jacobianMatrix2(ARA,ss,par):

    A=np.ones((len(ARA),ss.shape[1],3,3))*np.nan 

    X=ss[:,:,0]
    Y=ss[:,:,1]
    Z=ss[:,:,2]

    ARA=ARA[:,None]

    dxdx = -1 *X/X
    dxdy = 0  *X/X
    dxdz=-(((np.power(ARA,par['n_ARAX'])*(10**par['beta/alpha_X']-1))/ ( np.power(10**par['K_ARAX'],par['n_ARAX']) + np.power(ARA,par['n_ARAX']))+1)*par['n_ZX']*np.power((Z/10**(par['K_ZX'])),par['n_ZX']))
    dxdz=dxdz/(Z*np.power((np.power((Z/10**(par['K_ZX'])),par['n_ZX'])+1),2))

    dydx= - 10**par['beta/alpha_Y']*par['n_XY']*np.power((X/10**par['K_XY']),par['n_XY'])
    dydx=dydx/(X*np.power((np.power((X/10**par['K_XY']),par['n_XY'])+1),2))   
    dydy=-1 *Y/Y
    dydz= 0 *Y/Y

    dzdx = -(10**par['beta/alpha_Z']*par['n_XZ']*np.power((X/10**par['K_XZ']),par['n_XZ']))
    dzdx= dzdx /((np.power((Y/10**par['K_YZ']),par['n_YZ'])+1)*X*np.power((np.power(X/10**par['K_XZ'],par['n_XZ'])+1) ,2))

    dzdy= -(10**par['beta/alpha_Z']*par['n_YZ']*np.power((Y/10**par['K_YZ']),par['n_YZ']))
    dzdy= dzdy /((np.power((X/10**par['K_XZ']),par['n_XZ'])+1)*Y*np.power((np.power(Y/10**par['K_YZ'],par['n_YZ'])+1) ,2))
    dzdz = -1 *Z/Z

    A[:,:,0,0]=dxdx
    A[:,:,0,1]=dxdy
    A[:,:,0,2]=dxdz

    A[:,:,1,0]=dydx
    A[:,:,1,1]=dydy
    A[:,:,1,2]=dydz

    A[:,:,2,0]=dzdx
    A[:,:,2,1]=dzdy
    A[:,:,2,2]=dzdz

    return A

def approximateJacob(ARA,X,Y,Z,par):
    delta=10e-15
    #used to verify the Jacobain matrix. 

    x,y,z =Flow(X,Y,Z,ARA,par)

    dxdx = (Flow(X+delta,Y,Z,ARA,par)[0] - x)/delta
    dxdy = (Flow(X,Y+delta,Z,ARA,par)[0] - x)/delta
    dxdz = (Flow(X,Y,Z+delta,ARA,par)[0] - x)/delta

    dydx = (Flow(X+delta,Y,Z,ARA,par)[1] - y)/delta
    dydy = (Flow(X,Y+delta,Z,ARA,par)[1] - y)/delta
    dydz = (Flow(X,Y,Z+delta,ARA,par)[1] - y)/delta

    dzdx = (Flow(X+delta,Y,Z,ARA,par)[2] - z)/delta
    dzdy = (Flow(X,Y+delta,Z,ARA,par)[2] - z)/delta
    dzdz = (Flow(X,Y,Z+delta,ARA,par)[2] - z)/delta

    A=np.array(([dxdx,dxdy,dxdz],[dydx,dydy,dydz],[dzdx,dzdy,dzdz]))

    return A