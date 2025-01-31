#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from numpy import matlib
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.interpolate import BSpline
import random


def dictionarypoly(U,order,option=[]):
    #% Description: Construct the dictionary matrix phiX containing all multivariate monomials up to degree two for the Lorenz 96
    # % Input: U = [x1(t1) x2(t1) .... xn(t1)
    # %             x1(t2) x2(t2) .... xn(t2)
    # %                    ......
    # %             x1(tm) x2(tm) .... xn(tm)]
    # %        option = [] (monomial) or 'legendre'
    # % Output: the dictionary matrix phiX of size m by N, where m= #measurements and N = (n^2+3n+2)/2
    #if nargin ==1
    #if str(option) == 'mon' or str(option=='legendre):
    #end
    #U=np.array([[1,2,3],[4,5,6],[7,8,9]])
    if order == '2':
        m=int(np.size(U,0))
        n=int(np.size(U,1))
        phiX=np.zeros([m,int((n+1)*(n+2)/2)])
#         print('shape of dict',phiX.shape)
        phiX[:,0]=np.ones([1,m])
        phiX[:,1:n+1]=np.sqrt(3)*U
        for k in range(1,n+1):
            phiX[:,int(((k)*(2*(n)-k+3)/2)) : int(((k+1)*(n) -(k**2)/2 + k/2 +1))]=3*np.multiply(np.matlib.repmat(np.array([U[:,k-1]]).T,1,n+1-k),U[:,k-1:n])
            if option=='legendre':
                phiX[:, int((k*(2*n-k+3)/2))]=(np.sqrt(5)/2.0)*(3*np.array([np.square(U[:,k-1])]).T-np.ones([m,1])).T
          
    else:
        m=int(np.size(U,0))
        n=int(np.size(U,1))
        phiX=np.zeros([m,int((3*n**2 + 3*n + 2)/2)])
        phiX[:,0]=np.ones([1,m])
        phiX[:,1:n+1]=np.sqrt(3)*U
        for k in range(1,n+1):
            phiX[:,int(((k)*(2*(n)-k+3)/2)) : int(((k+1)*(n) -(k**2)/2 + k/2 +1 ))]=3*np.multiply(np.matlib.repmat(np.array([U[:,k
                                                                                                                               -1]]).T,1,n+1-k),U[:,k-1:n])
            phiX[:,int(((n+1)*(n) -(n**2)/2 + n/2 +1 )+(k-1)*n):int(((n+1)*(n) -(n**2)/2 + n/2 +1 )+(k)*n)]=3*np.sqrt(3)*np.multiply(np.matlib.repmat(np.array([U[:,k-1]**2]).T,1,n),U)
            if option=='legendre3':
                phiX[:, int((k*(2*n-k+3)/2))]=(np.sqrt(5)/2.0)*(3*np.array([np.square(U[:,k-1])]).T-np.ones([m,1])).T
                phiX[:, int(((n+1)*(n) -(n**2)/2 + n/2 +1 )+(k-1)*(n+1))]=(np.sqrt(7)/2.0)*(5*np.array([(U[:,k-1])**3]).T
                                                                                        - 3*np.array([(U[:,k-1])]).T).T

    return torch.from_numpy(phiX.astype('float32')).requires_grad_(True)


fro = 2
def error_rel(model_out1,model_out2,true_out1,true_out2):
    error1 = torch.norm(model_out1-true_out1,fro)/torch.norm(true_out1,fro)
    error2 = torch.norm(model_out2-true_out2,fro)/torch.norm(true_out2,fro)
    return error1, error2

def error_mat(model_out1,model_out2,true_out1,true_out2):
    error1 = torch.abs(model_out1-true_out1) #/model_out1.shape[0]
    error2 = torch.abs(model_out2-true_out2) #/model_out1.shape[0]
    return error1, error2

def company_data(company):
    if company == 'Syncrude':
        file_path = 'Syncrude2019-2023AllMonthly.xlsx'
        df_comp = pd.read_excel(file_path, sheet_name='Syncrude2019-2023')
    elif company == 'Suncor':
        file_path = 'SuncorMonthly2019-2023.xlsx'
        df_comp = pd.read_excel(file_path, sheet_name='SuncorAll2019-2023')
    else:
        file_path = 'Second_gen2019-2023.xlsx'
        df_comp = pd.read_excel(file_path, sheet_name='Second_gen2019-2023')
    return df_comp
        
    
def company_data(company):
    if company == 'Syncrude':
        file_path = 'Syncrude2019-2023AllMonthly.xlsx'
        df_comp = pd.read_excel(file_path, sheet_name='Syncrude2019-2023')
    elif company == 'Suncor':
        file_path = 'SuncorMonthly2019-2023.xlsx'
        df_comp = pd.read_excel(file_path, sheet_name='SuncorAll2019-2023')
    else:
        file_path = 'Second_gen2019-2023.xlsx'
        df_comp = pd.read_excel(file_path, sheet_name='Second_gen2019-2023')
    return df_comp
        
    
def year_days(year):
    if year == '2020':
        start = 366 
        end = 732
    elif year == '2021':
        start = 732 
        end = 1097
    elif year == '2022':
        start = 1097 
        end = 1463
    elif year == '2023':
        start = 1463 
        end = 1828
    elif year == '2020-2022':
        start = 366 
        end = 1463
    else:
        start = 366 
        end = 1828
    return start, end
        
    
def station(name,year):
    if name == 'MLSB':
        data_wbea = pd.read_csv('mildred_MLSB_daily_wbea2023.csv')
        if year == '2020':
            first = 9
            last = 141
        elif year == '2021':
            first = 141
            last = 265
        elif year == '2022':
            first = 265
            last = 348
        elif year == '2023':
            first = 348
            last = 623
        elif year =='2020-2022':
            first = 9
            last = 348
        else:
            first = 9
            last = 623
            
    elif name == 'Mannix':
        data_wbea = pd.read_csv('Mannix_daily_wbea.csv')
        if year == '2020':
            first = 150
            last = 350
        elif year == '2021':
            first = 350
            last = 492
        elif year == '2022':
            first = 492
            last = 653
        elif year == '2023':
            first = 653
            last = 832
        elif year =='2020-2022':
            first = 150
            last = 653
        else:
            first = 150
            last = 832
    elif name == 'Buffalo':
        data_wbea = pd.read_csv('Buffalo_wbea_2019-2023.csv')
        if year == '2020':
            first = 195
            last = 395
        elif year == '2021':
            first = 395
            last = 593
        elif year == '2022':
            first = 593
            last = 758
        elif year == '2023':
            first = 758
            last = 925
        elif year =='2020-2022':
            first = 195
            last = 758
        else:
            first = 195
            last = 925
            
    else:
        data_wbea = pd.read_csv('LowerCamp_daily_wbea2019-2023.csv')
        if year == '2020':
            first = 155
            last = 337
        elif year == '2021':
            first = 337
            last = 513
        elif year == '2022':
            first = 513
            last = 733
        elif year == '2023':
            first = 733
            last = 936
        elif year =='2020-2022':
            first = 155
            last = 733
        else:
            first = 155
            last = 936
        
    dataset = np.array(data_wbea)[first:last,3:]
    date_index = list(np.array(data_wbea)[first:last,1]-int(np.array(data_wbea)[first,1]))
    return dataset,date_index
        
