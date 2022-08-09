# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from JPN_Training_phase.config import args
criterion2 = nn.KLDivLoss()
from torch.utils.data import TensorDataset,DataLoader



class Small_Prediction(nn.Module):
    '''
    Sub Joint prediction model
    1.Predict possible drift categories(PrototypicalNet)
    2.Predict at which point the drift is likely to occur(DriftPointNet)
    '''
    def __init__(self,Data_Vector_Length):
        super(Small_Prediction,self).__init__()
        self.Data_Vector_Length = Data_Vector_Length
        self.loc_loss_fun = nn.MSELoss()
        self.Moduels = ModelNet(Data_Vector_Length=self.Data_Vector_Length)
        self.Task1 = DriftPointFNN(Data_Vector_Length=self.Data_Vector_Length)
        self.Task2 = DriftTypeFNN(Data_Vector_Length=self.Data_Vector_Length)
        self.AutomaticWeightedLoss= AutomaticWeightedLoss(2)
        self.TypematicWeightedLoss = AutomaticWeightedLoss(2)


    def forward(self,datax,datay,locy):
        #Get input data (center of mass)
        count = datax.shape[0]
        input = self.Moduels(datax)
        # input = datax
        pre_loc_y = self.Task1(input)
        pre_y2 = self.Task2(input)

        #drift type
        type_pred_S = torch.log_softmax(pre_y2, dim=-1)
        type_loss = F.nll_loss(type_pred_S, datay.long())

        # drift point
        loc_loss = self.loc_loss_fun(pre_loc_y, locy)


        loc_acc = self.cal_loc_acc(pre_loc_y, locy)

        type_acc = float(torch.sum((torch.argmax(type_pred_S, 1) == datay).float()) / count)

        # Combination of two Loss
        loss = self.AutomaticWeightedLoss(type_loss, loc_loss)
        return loss,type_loss,loc_loss,loc_acc,type_acc

    def cal_loc_dt_loss(self,Rs,Rt,y):
        '''
        cal Lb(Rs, Rt, y)
        if L2(Rs − yk) + m > L2(Rt − yk) -> Lb(Rs, Rt, y) = L2(Rt − yk)
        else Lb(Rs, Rt, y) = 0
        '''
        m = args.distillation_M
        rs_loss = self.loc_loss_fun(Rs, y)
        rt_loss = self.loc_loss_fun(Rt, y)
        if rs_loss + m > rt_loss:
            return rs_loss
        else:
            return 0


    def cal_loc_acc(self,pre_loc_y,locy):
        '''
        R**2 :cal loc acc
        where u is the residual sum of squares
         ((y_true - y_pred) ** 2).sum() and v is the total
            sum of squares ((y_true - y_true.mean()) ** 2).sum().
        '''
        u = torch.sum(torch.abs(locy - pre_loc_y) ** 2)
        v = torch.sum(torch.abs(locy - torch.mean(locy)) ** 2)
        acc_loc = 1 - u / v
        return acc_loc



class ModelNet(nn.Module):
    """
    network main function
    """
    def __init__(self,Data_Vector_Length):
        super(ModelNet, self).__init__()
        self.fc1 = nn.Linear(Data_Vector_Length, Data_Vector_Length)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

class DriftPointFNN(nn.Module):
    """
    normal feed-network
    """
    def __init__(self,Data_Vector_Length):
        super(DriftPointFNN, self).__init__()
        self.fc1 = nn.Linear(Data_Vector_Length, 300)
        # self.fc2 = nn.Linear(1000, Data_Vector_Length*2)
        # self.fc3 = nn.Linear(Data_Vector_Length*2, 1000)
        # self.fc4 = nn.Linear(1000, 500)
        # self.fc3 = nn.Linear(800, 200)
        # self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(300, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = self.dropout(x)
        x = self.fc5(x)
        return x

class DriftTypeFNN(nn.Module):
    """
    normal feed-network
    """
    def __init__(self,Data_Vector_Length):
        super(DriftTypeFNN, self).__init__()
        self.fc1 = nn.Linear(Data_Vector_Length, Data_Vector_Length*2)
        # self.fc2 = nn.Linear(Data_Vector_Length*2, Data_Vector_Length*2)
        # self.fc3 = nn.Linear(Data_Vector_Length*2, Data_Vector_Length)
        self.fc4 = nn.Linear(Data_Vector_Length*2, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def load_weights(filename, protonet, use_gpu):
    if use_gpu:
        protonet.load_state_dict(torch.load(filename))
    else:
        protonet.load_state_dict(torch.load(filename), map_location='cpu')
    return protonet

def init_lr_scheduler(optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=0.9,
                                           step_size=10)



