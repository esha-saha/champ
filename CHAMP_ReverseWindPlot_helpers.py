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


# In[2]:

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
        
    
def station(name,year,WD):
    if name == 'MLSB':
        file_path = 'Mildred_Lake_different_wind_dir.xlsx'
        data_wbea = pd.read_excel(file_path, sheet_name='Mildred_data_all')
        if year == '2020':
            first = 391
            last = 4915
        elif year == '2021':
            first = 4915
            last = 8945
        elif year == '2022':
            first = 8945
            last = 12582
        else:
            first = 12582
            last = 20153
    elif name == 'MLSB_red':
        file_path = 'Mildred_Lake_different_wind_dir_reducedch4.xlsx'
        data_wbea = pd.read_excel(file_path, sheet_name='Sheet1')
        if year == '2020':
            first = 391
            last = 4915
        elif year == '2021':
            first = 4915
            last = 8945
        elif year == '2022':
            first = 8945
            last = 12582
        else:
            first = 12582
            last = 20153
                
    elif name == 'Lower':
        file_path = 'Lower_different_wd.xlsx'
        data_wbea = pd.read_excel(file_path, sheet_name='Lower_camp_data_all')
        if year == '2020':
            first = 5936
            last = 13540 
        elif year =='2021':
            first = 13540
            last = 20200
        elif year == '2022':
            first = 20200
            last = 27550
        else:
            first = 27550
            last = 34723
            
    elif name == 'Mannix':
        file_path = 'Mannix_wind_dir.xlsx'
        data_wbea = pd.read_excel(file_path, sheet_name='Sheet1')
        if year == '2020':
            first = 5544
            last = 13320
        elif year =='2021':
            first = 13320
            last = 21132
        elif year == '2022':
            first = 21132
            last = 25820
        else:
            first = 25820
            last = 32592
            
    elif name == 'Buffalo':
        file_path = 'Buffalo_WD.xlsx'
        data_wbea = pd.read_excel(file_path, sheet_name='Buffalo_data_all')
        if year == '2020':
            first = 6631
            last = 13323
        elif year =='2021':
            first = 13323
            last = 19905
        elif year == '2022':
            first = 19905
            last = 27300
        else:
            first = 27300
            last = 35093
            
    else:
        print('dataset does not exist')
            
    
    dataset = np.array(data_wbea)[first:last,3:]
    print(dataset.shape)
#     dataset = data_wbea[first:last,3:]
    return dataset

        
        
neurons = 800
neurons2 = 200
n_dims = 1
n_dims1 = 34+1-19 #20+1
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.lin1 = nn.Linear(34,neurons,bias = True)
        self.lin2 = nn.Linear(neurons,neurons,bias = True)
        self.lin22 = nn.Linear(neurons,neurons,bias = True)
        self.lin222 = nn.Linear(neurons,neurons,bias = True)
        self.lin3 = nn.Linear(neurons,n_dims,bias = True)
        self.lin4 = nn.Linear(n_dims1,neurons2,bias = True)
        self.lin5 = nn.Linear(neurons2,neurons2,bias = True)
        self.lin6 = nn.Linear(neurons,neurons,bias = True)
        self.lin7 = nn.Linear(neurons2,n_dims,bias = True)

    def forward(self, x,t,true_ch4):
        y = self.lin1(x[:,:]).relu()
        y = self.lin2(y).relu()
        y = self.lin3(y)
        ch4_conc = y.abs()
        data_conc = torch.cat((true_ch4,x[:,:15]),dim = 1)
        z = self.lin4(data_conc).relu()
        z = self.lin5(z).relu()
        z = self.lin7(z)
        ch4_emm = z.abs()
        return ch4_conc,ch4_emm

# net = Net()

def get_emm_wd(company,station_name,year,WD,a,b,wd_index):
    df_syncrude = company_data(company) 
    dataset_full = station(station_name,year,WD)

    index = [i for i in range(len(dataset_full[:,wd_index])) if dataset_full[i,wd_index] >a and dataset_full[i,wd_index] <b or dataset_full[i,wd_index]==b]
    tot_time = len(index)

    conc_data = (dataset_full[index,0]).astype('float32')
    wbea_data = (dataset_full[index,1:]).astype('float32')


    plt.plot(conc_data)

    diluent2023 = np.ones([wbea_data.shape[0],19])
    wbea_pars2023 = torch.tensor(wbea_data)
    ch4_wbea2023 = conc_data
    input_data2023 = torch.concatenate((wbea_pars2023,torch.tensor(diluent2023.astype('float32'))),dim = 1) # uses real or 2nd gen diluent data
    print(input_data2023.shape)
    out_min, out_max, ch4_min, ch4_max, inp_min, inp_max = scale_par(station_name)
    input_data2023 = (input_data2023 - inp_min)/(inp_max-inp_min)


    #create output data for concentrations taken from real data from WBEA
    ch4_conc_data2023 = torch.from_numpy(ch4_wbea2023.reshape(-1,1).astype('float32')).reshape(-1,1)

    ch4_conc_data2023 = (ch4_conc_data2023 - out_min)/(out_max-out_min)

    time_data2023 = np.ones(ch4_conc_data2023.shape[0]).reshape(-1,1) #np.arange(1, len(input_data)+1).reshape(-1, 1)

    time_data2023 = torch.tensor(time_data2023, dtype=torch.float32)
    scale_time_data2023 = torch.norm(time_data2023)
    time_data2023 /= scale_time_data2023

    input_data2023 = input_data2023.requires_grad_(True)
    time_data2023 = time_data2023.requires_grad_(True)


    if station_name == 'MLSB':
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse_MLSB_syncrude_2020-2022')
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse2_MLSB_syncrude_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse4_MLSB_syncrude_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
    elif station_name == 'MLSB_red':
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse_MLSB_syncrude_2020-2022')
        
#         ch4_min = 0.144
#         ch4_max = 0.2666
    elif station_name == 'Buffalo':
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse_MLSB_syncrude_2020-2022')
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse_Mannix_suncor_2020-2022')
#         ch4_min = 0.144
#         ch4_max = 0.2666
    elif station_name == 'Mannix':
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse_Mannix_suncor_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse2_Mannix_suncor_2020-2022')
#         ch4_min = 0.1005
#         ch4_max = 0.17
    else:
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
#         ch4_min = 0.1
#         ch4_max = 0.17

    _, emm_2023_champ = model_reverse_trace(input_data2023, time_data2023,ch4_conc_data2023)
    emm2023_champ_unscaled = emm_2023_champ #*(ch4_max-ch4_min) + ch4_min
    
    return (np.sum(emm2023_champ_unscaled.detach().numpy())/tot_time)*365


def filtered_station(station_name,year,pos):
    if station_name == 'Lower':
        sheet_dict = pd.ExcelFile('Lower_filtered_data.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2020':
            df = pd.read_excel('Lower_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2020') & (df['date'] < '31-12-2020')].copy()
        elif year == '2021':
            df = pd.read_excel('Lower_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2021') & (df['date'] < '31-12-2021')].copy()
        elif year == '2022':
            df = pd.read_excel('Lower_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2022') & (df['date'] < '31-12-2022')].copy()
        else: 
            df = pd.read_excel('Lower_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
            
    elif station_name == 'Mannix':
        sheet_dict = pd.ExcelFile('Mannix_filtered_data.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2020':
            df = pd.read_excel('Mannix_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2020') & (df['date'] < '31-12-2020')].copy()
        elif year == '2021':
            df = pd.read_excel('Mannix_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2021') & (df['date'] < '31-12-2021')].copy()
        elif year == '2022':
            df = pd.read_excel('Mannix_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2022') & (df['date'] < '31-12-2022')].copy()
        else: 
            df = pd.read_excel('Mannix_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
            
    elif station_name == 'Buffalo':
        sheet_dict = pd.ExcelFile('Buffalo_filtered_data.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2020':
            df = pd.read_excel('Buffalo_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2020') & (df['date'] < '31-12-2020')].copy()
        elif year == '2021':
            df = pd.read_excel('Buffalo_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2021') & (df['date'] < '31-12-2021')].copy()
        elif year == '2022':
            df = pd.read_excel('Buffalo_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2022') & (df['date'] < '31-12-2022')].copy()
        else: 
            df = pd.read_excel('Buffalo_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
            
    elif station_name == 'MLSB':
        sheet_dict = pd.ExcelFile('Mildred_filtered_data.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2020':
            df = pd.read_excel('Mildred_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2020') & (df['date'] < '31-12-2020')].copy()
        elif year == '2021':
            df = pd.read_excel('Mildred_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2021') & (df['date'] < '31-12-2021')].copy()
        elif year == '2022':
            df = pd.read_excel('Mildred_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2022') & (df['date'] < '31-12-2022')].copy()
        else: 
            df = pd.read_excel('Mildred_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
    elif station_name == 'MLSB_red':
        sheet_dict = pd.ExcelFile('MLSB_reduce_filtered_data.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2023':
            df = pd.read_excel('MLSB_reduce_filtered_data.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
#     print(filtered_df)
    elif station_name == 'Mannix_red':
        sheet_dict = pd.ExcelFile('Mannix_filtered_reduced_shifted.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2023':
            df = pd.read_excel('Mannix_filtered_reduced_shifted.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
    elif station_name == 'Lower_red':
        sheet_dict = pd.ExcelFile('Lower_filtered_reduced_shifted.xlsx')
        name = sheet_dict.sheet_names[pos]
        if year == '2023':
            df = pd.read_excel('Lower_filtered_reduced_shifted.xlsx',sheet_name = name)
            filtered_df = df[(df['date'] >= '01-01-2023') & (df['date'] < '31-12-2023')].copy()
#     print(filtered_df)
    return filtered_df
  
def inter(interpolate,xdays_true,dataset_true):
    if interpolate == True:
        x = np.array(xdays_true)
        print('actual true points are',x.shape)
        
#         plt.figure(figsize=(4, 3))
#         plt.plot(x,dataset_true[:-1,0],'-*')
        
        y = np.zeros([dataset_true.shape[0],dataset_true.shape[1]])
        xx = np.linspace(0,365-1,365)
        dataset_inter = np.zeros((365,dataset_true.shape[1]))
        for i in range(dataset_true.shape[1]):
            y[:,i] = np.array(dataset_true)[:,i]
    #         dataset_inter[:,i] = np.interp(xx, x, y[:,i])
        spl = BSpline(x, y, 1,extrapolate = 'periodic')
        dataset_inter = spl(xx) 

        dataset = dataset_inter
        xdays = list(np.arange(len(xx))[:])
    else:
        dataset = dataset_true
        xdays = xdays_true
#     plt.plot(dataset_true[:,0])
    return xdays, dataset

def scale_par(station):
    if station == 'MLSB':
        conc_min = 1.8932 
        conc_max = 3.4530
        emm_min = 0.1440
        emm_max = 0.2666
        inp_min = torch.tensor([-2.9500e+01,  0.0000e+00,  1.8467e+01,  0.0000e+00,  2.7110e+02,
         9.1000e+00,  5.0000e-01,  7.0000e-01,  3.4498e+02, -3.5035e+01,
         0.0000e+00,  0.0000e+00,  0.0000e+00, -1.2333e+00,  2.0000e-01,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([2.7650e+01, 1.1600e+01, 9.9800e+01, 5.5350e+01, 3.0960e+02, 1.0000e+02,
        3.5800e+01, 8.3000e+00, 1.0005e+03, 1.7375e+01, 1.5090e+03, 8.0200e+02,
        7.7435e+02, 6.0000e-01, 3.9000e+00, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
    elif station == 'MLSB_red':
        conc_min = 1.8932 
        conc_max = 3.4530
        emm_min = 0.1440
        emm_max = 0.2666
        inp_min = torch.tensor([-2.9500e+01,  0.0000e+00,  1.8467e+01,  0.0000e+00,  2.7110e+02,
         9.1000e+00,  5.0000e-01,  7.0000e-01,  3.4498e+02, -3.5035e+01,
         0.0000e+00,  0.0000e+00,  0.0000e+00, -1.2333e+00,  2.0000e-01,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([2.7650e+01, 1.1600e+01, 9.9800e+01, 5.5350e+01, 3.0960e+02, 1.0000e+02,
        3.5800e+01, 8.3000e+00, 1.0005e+03, 1.7375e+01, 1.5090e+03, 8.0200e+02,
        7.7435e+02, 6.0000e-01, 3.9000e+00, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
    elif station == 'Mannix':
        conc_min = 1.8722 
        conc_max = 4.5140
        emm_min = 0.1005
        emm_max = 0.1702
        inp_min = torch.tensor([-3.2925e+01,  0.0000e+00,  1.9500e+01,  0.0000e+00, -1.0778e+00,
         1.3333e-01,  2.8010e+02,  7.7000e+00,  7.0000e-01,  8.0000e-01,
         7.7621e+02, -4.0170e+01,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([3.0900e+01, 1.5900e+01, 9.8533e+01, 2.5000e+01, 3.0000e-01, 3.7357e+00,
        3.3990e+02, 9.6100e+01, 3.3393e+01, 8.8214e+00, 1.0066e+03, 2.0300e+01,
        1.5750e+03, 8.2600e+02, 7.4755e+02, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
    elif station == 'Mannix_red':
        conc_min = 1.8722 
        conc_max = 4.5140
        emm_min = 0.1005
        emm_max = 0.1702
        inp_min = torch.tensor([-3.2925e+01,  0.0000e+00,  1.9500e+01,  0.0000e+00, -1.0778e+00,
         1.3333e-01,  2.8010e+02,  7.7000e+00,  7.0000e-01,  8.0000e-01,
         7.7621e+02, -4.0170e+01,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([3.0900e+01, 1.5900e+01, 9.8533e+01, 2.5000e+01, 3.0000e-01, 3.7357e+00,
        3.3990e+02, 9.6100e+01, 3.3393e+01, 8.8214e+00, 1.0066e+03, 2.0300e+01,
        1.5750e+03, 8.2600e+02, 7.4755e+02, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
    elif station == 'Buffalo':
        conc_min = 1.8932 
        conc_max = 3.4530
        emm_min = 0.1440
        emm_max = 0.2666
        inp_min = torch.tensor([-2.9500e+01,  0.0000e+00,  1.8467e+01,  0.0000e+00,  2.7110e+02,
         9.1000e+00,  5.0000e-01,  7.0000e-01,  3.4498e+02, -3.5035e+01,
         0.0000e+00,  0.0000e+00,  0.0000e+00, -1.2333e+00,  2.0000e-01,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([2.7650e+01, 1.1600e+01, 9.9800e+01, 5.5350e+01, 3.0960e+02, 1.0000e+02,
        3.5800e+01, 8.3000e+00, 1.0005e+03, 1.7375e+01, 1.5090e+03, 8.0200e+02,
        7.7435e+02, 6.0000e-01, 3.9000e+00, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
        
    elif station == 'Lower_red':
        conc_min = 1.8690 
        conc_max = 2.9200
        emm_min = 0.1005
        emm_max = 0.1702
        inp_min = torch.tensor([-3.5200e+01,  0.0000e+00,  2.1800e+01,  0.0000e+00,  1.4000e+02,
         6.6250e+00,  1.0000e-01,  5.0000e-01,  9.3900e+02, -3.9090e+01,
         0.0000e+00,  0.0000e+00, -9.0000e-01,  2.0000e-01,  0.0000e+00,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([2.8155e+01, 9.7143e+00, 9.8767e+01, 8.9300e+01, 1.9820e+02, 1.0650e+02,
        2.8580e+01, 9.1000e+00, 1.0045e+03, 1.6680e+01, 1.5700e+03, 7.9930e+02,
        9.0000e-01, 3.3000e+00, 8.1700e+02, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
        
        
    
    else:
        conc_min = 1.8690 
        conc_max = 2.9200
        emm_min = 0.1005
        emm_max = 0.1702
        inp_min = torch.tensor([-3.5200e+01,  0.0000e+00,  2.1800e+01,  0.0000e+00,  1.4000e+02,
         6.6250e+00,  1.0000e-01,  5.0000e-01,  9.3900e+02, -3.9090e+01,
         0.0000e+00,  0.0000e+00, -9.0000e-01,  2.0000e-01,  0.0000e+00,
         1.9930e-01,  1.3134e-01,  5.0385e-02,  2.4279e-02,  1.5661e-01,
         1.8470e-01,  5.4107e-02,  7.5979e-03,  6.0933e-02,  7.0775e-02,
         1.9414e-01,  7.3393e-02,  4.0766e-02,  5.4472e-02,  9.8205e-03,
         1.6400e-02,  2.8326e-03,  1.6000e-01,  9.0245e+00])
        inp_max = torch.tensor([2.8155e+01, 9.7143e+00, 9.8767e+01, 8.9300e+01, 1.9820e+02, 1.0650e+02,
        2.8580e+01, 9.1000e+00, 1.0045e+03, 1.6680e+01, 1.5700e+03, 7.9930e+02,
        9.0000e-01, 3.3000e+00, 8.1700e+02, 3.5477e-01, 2.3379e-01, 8.9687e-02,
        4.1921e-02, 2.7040e-01, 3.1890e-01, 9.3422e-02, 1.3119e-02, 9.9548e-02,
        1.1563e-01, 3.1717e-01, 1.1990e-01, 6.6601e-02, 8.8993e-02, 1.6044e-02,
        2.7868e-02, 1.8876e-02, 7.3094e-01, 4.0285e+01])
#         print(inp_max.shape)
    return conc_min, conc_max, emm_min, emm_max,inp_min, inp_max
    
        
    

def get_emm_wd_true(company,station_name,year,WD,pos):
    df_syncrude = company_data(company)
    year_filtered = filtered_station(station_name,year,pos)
#     print(year_filtered)
    differ = np.zeros(len(year_filtered))
    final_xday = np.zeros(len(year_filtered)-1)
    for i in range(len(year_filtered)-1):
        differ[i+1] = (year_filtered.iloc[i+1,0] - year_filtered.iloc[i,0]).days
        final_xday[i] = np.sum(differ[:i+1])
    xdays_final,dataset_full = inter(True,final_xday,np.array(year_filtered)[:,3:])
#     sheets_dict = pd.read_excel('Lower_filtered_data.xlsx', sheet_name=None)
 
    conc_data = (dataset_full[:,0]).astype('float32')
    wbea_data = (dataset_full[:,1:]).astype('float32')
#     print(conc_data[:3],wbea_data[:3,:5])

#     print(dataset_full.shape,conc_data.shape,wbea_data.shape)
#     plt.figure(figsize=(4, 3))
#     plt.plot(conc_data)
#     plt.plot(wbea_data[:,4])


    # ## Trace source through reverse modeling

    # TEST FOR 2023

    # In[5]:


    diluent2023 = np.ones([wbea_data.shape[0],19])
#     print(diluent_full.shape)
    wbea_pars2023 = torch.tensor(wbea_data)
    ch4_wbea2023 = conc_data
    input_data2023 = torch.concatenate((wbea_pars2023,torch.tensor(diluent2023.astype('float32'))),dim = 1) # uses real or 2nd gen diluent data
    print(wbea_pars2023.shape)
    #scale the input data
    out_min, out_max, ch4_min, ch4_max, inp_min, inp_max = scale_par(station_name)
    print(input_data2023.shape,inp_min.shape)

#     inp_min = torch.min(input_data2023,dim=0)[0]
#     inp_max = torch.max(input_data2023,dim=0)[0]
    input_data2023 = (input_data2023 - inp_min)/(inp_max-inp_min)


    #create output data for concentrations taken from real data from WBEA
    ch4_conc_data2023 = torch.from_numpy(ch4_wbea2023.reshape(-1,1).astype('float32')).reshape(-1,1)



    # ch4_conc_data/=scale_out
#     out_min = 1.86 #torch.min(ch4_conc_data2023)
#     out_max = 2.9 #torch.max(ch4_conc_data2023)
    ch4_conc_data2023 = (ch4_conc_data2023 - out_min)/(out_max-out_min)
    # plt.plot(ch4_conc_data2023)

    time_data2023 = np.ones(ch4_conc_data2023.shape[0]).reshape(-1,1) #np.arange(1, len(input_data)+1).reshape(-1, 1)

    # Convert time_data to a PyTorch tensor and normalize
    time_data2023 = torch.tensor(time_data2023, dtype=torch.float32)
    scale_time_data2023 = torch.norm(time_data2023)
    time_data2023 /= scale_time_data2023

    input_data2023 = input_data2023.requires_grad_(True)
    time_data2023 = time_data2023.requires_grad_(True)


    # In[6]:


    

#     net = Net()


    # In[7]:


    if station_name == 'MLSB':
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse_MLSB_syncrude_2020-2022')
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse2_MLSB_syncrude_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse4_MLSB_syncrude_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse_Mannix_suncor_2020-2022')
    elif station_name == 'MLSB_red':
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse_MLSB_syncrude_2020-2022')
#         ch4_min = 0.144
#         ch4_max = 0.2666
    elif station_name == 'Buffalo':
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/MildredLake/net_reverse2_MLSB_syncrude_2020-2022')
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse_Mannix_suncor_2020-2022')
#         ch4_min = 0.144
#         ch4_max = 0.2666
    elif station_name == 'Mannix':
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse1_Mannix_suncor_2020-2022')
    elif station_name == 'Mannix_red':
#         model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/Mannix/net_reverse1_Mannix_suncor_2020-2022')
#         ch4_min = 0.1
    elif station_name == 'Lower_red':
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
    else:
        model_reverse_trace = torch.load('/Users/eshasaha/Desktop/ILMEE Postdoc Research Projects/NSERC Alliance P1-P2/HybridModel/Results/LowerCamp/net_reverse_LowerCamp_suncor_2020-2022')
#         ch4_min = 0.1
#         ch4_max = 0.17

    _, emm_2023_champ = model_reverse_trace(input_data2023, time_data2023,ch4_conc_data2023)
    emm2023_champ_unscaled = emm_2023_champ*(ch4_max-ch4_min) + ch4_min
    # emm_data2023_true = ch4_emm_data2023*(ch4_max-ch4_min) + ch4_min


    # In[8]:


    # plt.plot(emm_data2023_true,label = 'True')
#     plt.plot(emm2023_champ_unscaled.detach().numpy(),label = 'Predicted')
#     plt.legend()


    # In[9]:


#     print(np.sum(emm2023_champ_unscaled.detach().numpy())) #,np.sum(emm_data2023_true.detach().numpy()))
    
    return np.sum(emm2023_champ_unscaled.detach().numpy())*16.04





