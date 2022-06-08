import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.decomposition import PCA
import random
import os
from sklearn.preprocessing import StandardScaler



import model_equation as meq_2
import model_equation_1_ind as meq_1

parlist_2_ind= meq_2.parlist
parlist_1_ind = meq_1.parlist
filename_2ind= os.path.dirname(os.path.abspath(__file__))+"/ACDC_abcsmc_2_2022-03-24"
filename_1ind = os.path.dirname(os.path.abspath(__file__))+"/ACDC_abcsmc_1_2022-03-30"



def load(number,filename,parlist):
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



###########
#Add generation list for each experiment here

n_1=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']

n_2=['final','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

######

def PCA_analysis(gen_list,filename,parlist,npar=5000):   #npar: number of randomly selected paramter set from each generation

    #Create concatenated dataset of all generations
    for i_num,i in enumerate(gen_list):
        p, pdf= load(i,filename,parlist)
        pdf.drop(['dist'],axis=1,inplace=True) #remove distance column
        ran_ind = random.sample(range(0,len(p)),npar)
        pd_new = pdf.iloc[ran_ind]
        if i_num == 0: #if the first dataset
            pd_final = pd_new
        else:
            pd_final = pd.concat([pd_new,pd_final],ignore_index=True)

    scalar = StandardScaler()
    pd_final = pd.DataFrame(scalar.fit_transform(pd_final), columns=pd_final.columns)


    par_pca = PCA(n_components=len(pd_final.columns))
    components = par_pca.fit(pd_final).components_
    components = pd.DataFrame(components).transpose()
    components.columns = ['PC{}'.format(str(n)) for n in range(1,len(pd_final.columns)+1)]
    components.index  = pd_final.columns
    
    max_weighting_dict = dict.fromkeys(components.columns)

    
    for i,PC in enumerate(components.columns):
        absolute_comp = components[PC].abs()
        max_weighting_dict[PC]=  components[PC][absolute_comp.max()==absolute_comp] #gives parameter with maximum weighting for each PC
    



    var_ratio = par_pca.explained_variance_ratio_
    var_ratio = pd.DataFrame(var_ratio)
    var_ratio.index = [n for n in range(1,len(pd_final.columns)+1)]
    var_ratio.columns = ['Proportion of variance']


    

    #Consider adding a knee detecting algorithm/function to automate the essential PC identification process

    return max_weighting_dict,components,var_ratio

def PCA_plot(components,var_ratio,filename):

    #Bar plot
    for i,PC in enumerate(components.columns):
        #Unsorted weighting
        fig =  plt.figure(figsize=(20,15))
        ax=plt.subplot()
        plt.bar(range(len(components)),components[PC],color='navy')
        plt.ylim(-1,1)

        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components.index)
        ax.set_yticks(np.arange(-1,1,0.1))
        ax.set_yticklabels(np.round(np.arange(-1,1,0.1),decimals=1))


        plt.title('Loading scores for Principle Component {}'.format(i+1),size = 25)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=20)
        plt.xlabel('Parameters',size=20)
        plt.ylabel('Weightings',size=20)
        plt.savefig(filename+'/PCA/Final/'+'Loading_score_{}.png'.format(PC),dpi=300)
        plt.clf()


        sort_df  = components[PC].abs().sort_values(ascending=False)

        #Percentage horizontal barplot
        percent = sort_df.values/sort_df.values.sum()        
        fig =  plt.figure(figsize=(18,10))
        ax=plt.subplot()
        plt.barh(range(len(sort_df)),percent,color='orange')
        plt.xlim(0,0.3)


        ax.set_yticks(range(len(sort_df)))
        ax.set_yticklabels(sort_df.index)
        ax.set_xticks(np.arange(0,0.3,0.05))
        ax.set_xticklabels(np.round(np.arange(0,30,5)))


        plt.title('Principle Component {}'.format(i+1),size = 30)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

        plt.ylabel('Parameters',size=30)
        plt.xlabel('Fraction of PC explained (%)',size=30)
        plt.savefig(filename+'/PCA/Final/'+'Fraction of {} explained by each parameter.png'.format(PC))

        '''
        #Sorted absolute weighting


        
        fig =  plt.figure(figsize=(23,15))
        ax=plt.subplot()
        plt.bar(range(len(sort_df)),sort_df,color='navy')
        plt.ylim(0,1)


        ax.set_xticks(range(len(sort_df)))
        ax.set_xticklabels(sort_df.index)
        ax.set_yticks(np.arange(0,1,0.1))
        ax.set_yticklabels(np.round(np.arange(0,1,0.1),decimals=1))


        plt.title('Sorted Loading scores for Principle Component {}'.format(i+1),size = 25)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=20)

        plt.xlabel('Parameters',size=20)
        plt.ylabel('Absolute Weightings',size=20)
        plt.savefig(filename+'/PCA/Final/'+'Sorted_loading_score_{}.png'.format(PC),dpi=300)
        plt.clf()

        '''
        '''
        #Pie chart
        fig =  plt.figure(figsize=(16,12))
        plt.pie(sort_df,labels =sort_df.index, textprops={'fontsize': 15})
        plt.title('Fraction of {} explained by each parameter'.format(PC),size=15)
        plt.savefig(filename+'/PCA/Final/'+'Fraction of {} explained by each parameter.png'.format(PC))
        '''

    #Scree plot
    '''
    fig= plt.figure(figsize=(16,12))

    plt.xticks(var_ratio.index,fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(var_ratio,marker='o',markersize=10)
    plt.title('Proportion of variance ',fontsize=25)
    plt.xlabel('Principle components',fontsize=20)
    plt.ylabel('Explained variance ratio',fontsize=20)


    plt.savefig(filename+'/PCA/Final/'+'Scree plot',dpi=300)
    '''







def compare_PCA(filename_1,filename_2,parlist_1,parlist_2,genlist_1,genlist_2,npar=1000):

    #Creating dataset for both hypotheses
    #1 inducer
    for i_num,i in enumerate(genlist_1):
        p, pdf= load(i,filename_1,parlist_1)
        pdf.drop(['dist'],axis=1,inplace=True)
        ran_ind = random.sample(range(0,len(p)),npar)
        pd_new = pdf.iloc[ran_ind]
        if i_num == 0:
            pd_final_1 = pd_new
        else:
            pd_final_1 = pd.concat([pd_new,pd_final_1],ignore_index=True)

    scalar = StandardScaler()
    pd_final_1 = pd.DataFrame(scalar.fit_transform(pd_final_1), columns=pd_final_1.columns)

    #2 inducers
    for i_num,i in enumerate(genlist_2):
        p, pdf= load(i,filename_2,parlist_2)
        pdf.drop(['dist'],axis=1,inplace=True)
        ran_ind = random.sample(range(0,len(p)),npar)
        pd_new = pdf.iloc[ran_ind]
        if i_num == 0:
            pd_final_2 = pd_new
        else:
            pd_final_2 = pd.concat([pd_new,pd_final_2],ignore_index=True)
    pd_final_2 = pd.DataFrame(scalar.fit_transform(pd_final_2), columns=pd_final_2.columns)


    #PCA and obtain components weighting dataframe
    twoind_pca = PCA(n_components=len(pd_final_2.columns))
    two_components =  twoind_pca.fit(pd_final_2).components_
    two_components = pd.DataFrame(two_components).transpose()
    two_components.columns = ['PC{}'.format(str(n)) for n in range(1,len(pd_final_2.columns)+1)]
    two_components.index  = pd_final_2.columns


    oneind_pca = PCA(n_components=len(pd_final_1.columns))
    one_components =  oneind_pca.fit(pd_final_1).components_
    one_components = pd.DataFrame(one_components).transpose()
    one_components.columns = ['PC{}'.format(str(n)) for n in range(1,len(pd_final_1.columns)+1)]
    one_components.index  = pd_final_1.columns[0:len(pd_final_1.columns)]


    one_components= one_components.reindex(two_components.index,fill_value=0) #Set one inducers component dataframe to be same size as two inducers


    #Unsorted weighting bar plot:
    x_axis = 2.5*np.arange(len(two_components.index))

    for i,PC in enumerate(one_components.columns):

        fig =  plt.figure(figsize=(18,12))

        plt.bar(x_axis +1,two_components[PC],width=1.0,label='Two inducers',color='navy')
        plt.bar(x_axis +1*2,one_components[PC],width=1.0, label='One inducers',color='Orange')

        plt.xticks(x_axis+1.5,two_components.index,fontsize=10)
        plt.ylim(-1,1)
        plt.yticks(np.arange(-1,1,0.1),np.round(np.arange(-1,1,0.1),decimals=1))
        plt.ylabel('Weightings',fontsize=15)
        plt.xlabel('Parameters',fontsize=15)
        plt.title('Weighting scores at principle component {}'.format(str(i+1)),fontsize=18)
        plt.legend(fontsize=15)

        #plt.savefig(filename_2+'/1ind_vs_2ind'+'/PCA/Final/'+'Loading_score_{}.png'.format(PC),dpi=300)


    #Sorted weighting bar plot:
        sort2_df  = two_components[PC].abs().sort_values(ascending=False)
        sort1_df  = one_components[PC].reindex(sort2_df.index).abs()
        percent_2 = sort2_df.values/sort2_df.values.sum()        
        percent_1 = sort1_df.values/sort1_df.values.sum()        
        fig =  plt.figure(figsize=(20,12))

        plt.barh(x_axis +1,percent_2,label='Two inducers',color='navy')
        plt.barh(x_axis +1*2,percent_1,label='One inducer',color='Orange')

        plt.yticks(x_axis+1.5,sort2_df.index,fontsize=10)
        plt.xlim(0,0.3)
        plt.xticks(np.arange(0,0.3,0.05),np.round(np.arange(0,0.3,0.05),decimals=2))
        plt.legend(fontsize=15)
        plt.xlabel('Fraction of PC explained (%)',fontsize=20)
        plt.ylabel('Parameters',fontsize=15)
        plt.title('Sorted absolute weightings for principle component {}'.format(i+1),fontsize=18)
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

        plt.savefig(filename_2+'/1ind_vs_2ind'+'/PCA/Final/'+'Sorted_loading_score_{}.png'.format(PC),dpi=300)




    #Explained variance ratio for each PC
    var_ratio_2 = twoind_pca.explained_variance_ratio_ 
    var_ratio_2 = pd.DataFrame(var_ratio_2)
    var_ratio_2.index = [n for n in range(1,len(pd_final_2.columns)+1)]
    var_ratio_2.columns = ['Proportion of variance']


    var_ratio_1 = oneind_pca.explained_variance_ratio_ 
    var_ratio_1 = pd.DataFrame(var_ratio_1)
    var_ratio_1.index = [n for n in range(1,len(pd_final_1.columns)+1)]
    var_ratio_1.columns = ['Proportion of variance']







    #Scree plot
    plt.figure(figsize=(16,12))
    plt.plot(var_ratio_1,marker='o',label='one inducer',markersize=10,color='Orange')
    plt.plot(var_ratio_2,marker='^',label='two inducer',markersize=10,color='Navy')
    plt.yticks(fontsize=20)
    plt.xticks(range(1,len(var_ratio_2)+1),fontsize=20)
    plt.xlabel('Principle components',fontsize=25)
    plt.ylabel('Explained variance ratio',fontsize=25)
    plt.title('Proportion of variance',fontsize=30)
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    plt.legend(fontsize=15, prop={'size': 20})

    plt.savefig(filename_2+'/1ind_vs_2ind'+'/PCA/Final/'+'Scree plot',dpi=300)

    return two_components,one_components,var_ratio_2,var_ratio_1
        
















######



#max_weighting_dict_2,components_2,var_ratio_2= PCA_analysis(n_2,filename_2ind,parlist_2_ind)


#PCA_plot(components_2,var_ratio_2,filename_2ind)





#PCA_compare = compare_PCA(filename_1ind,filename_2ind,parlist_1_ind,parlist_2_ind,n_1,n_2)

#for final generation only

#max_weighting_dict_2,components_2,var_ratio_2= PCA_analysis(['final'],filename_2ind,parlist_2_ind)
#PCA_plot(components_2,var_ratio_2,filename_2ind)
#print(var_ratio_2)

max_weighting_dict_1,components_1,var_ratio_1= PCA_analysis(['final'],filename_1ind,parlist_1_ind)
#PCA_plot(components_1,var_ratio_1,filename_1ind)

#print(var_ratio_1.values.sum())
#compare_PCA(filename_1ind,filename_2ind,parlist_1_ind,parlist_2_ind,['final'],['final'],npar=5000)