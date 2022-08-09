# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from JPN_Training_phase.config import args
criterion2 = nn.KLDivLoss()



class Sub_Joint_Prediction(nn.Module):
    '''
    Sub Joint prediction model
    1.Predict possible drift categories(PrototypicalNet)
    2.Predict at which point the drift is likely to occur(DriftPointNet)
    '''
    def __init__(self,Data_Vector_Length):
        super(Sub_Joint_Prediction,self).__init__()
        self.Data_Vector_Length = Data_Vector_Length
        self.embedd_layer = nn.Linear(Data_Vector_Length, Data_Vector_Length)
        self.loc_loss_fun = nn.MSELoss()
        self.KLDivLoss_fun = nn.KLDivLoss()
        self.drift_type_distillation = DriftTypeFNN(Data_Vector_Length=self.Data_Vector_Length)
        self.drift_point_distillation = DriftPointFNN(Data_Vector_Length = self.Data_Vector_Length)

        self.AutomaticWeightedLoss= AutomaticWeightedLoss(2)


    def forward(self,datax,typey,locy,type_pred_T,loc_pred_T,loc_W):
        count = datax.shape[0]
        pred_type = self.drift_type_distillation(datax)
        pred_loc ,loss_w= self.drift_point_distillation(datax,loc_W)
        #drift type loss
        type_loss = self.cal_type_loss(pred_type,typey,type_pred_T)
        # drift point loss
        loc_loss = self.cal_loc_loss(pred_loc,loc_pred_T,locy,loss_w)
        # Acc of drift type and point
        loc_acc = self.cal_loc_acc(pred_loc, locy)
        type_acc = float(torch.sum((torch.argmax(pred_type, 1) == typey).float()) / count)
        # Combination of two Loss
        loss = self.AutomaticWeightedLoss(type_loss, loc_loss)
        return loss,type_loss,loc_loss,loc_acc,type_acc


    def cal_type_loss(self,pred_type,datay,type_pred_T):
        '''
        cal Lb(Rs, Rt, y)
        if L2(Rs − yk) + m > L2(Rt − yk) -> Lb(Rs, Rt, y) = L2(Rt − yk)
        else Lb(Rs, Rt, y) = 0
        '''
        pre_type = torch.log_softmax(pred_type / args.distillation_T, dim=-1)
        type_loss1 = F.nll_loss(pre_type, datay.long())  # learning student_model(real target)
        # learning teacher_model(soft target)
        type_loss2 = self.KLDivLoss_fun(pre_type, type_pred_T) * args.distillation_T * args.distillation_T
        type_loss = type_loss1 * (1 - args.distillation_type_alpha) + type_loss2 * args.distillation_type_alpha

        return type_loss

    def cal_loc_loss(self,pred_loc,loc_pred_T,locy,loss_w):
        '''
        cal Lb(Rs, Rt, y)
        if L2(Rs − yk) + m > L2(Rt − yk) -> Lb(Rs, Rt, y) = L2(Rt − yk)
        else Lb(Rs, Rt, y) = 0
        '''
        rs_loss = self.loc_loss_fun(pred_loc, locy)
        rt_loss = self.loc_loss_fun(loc_pred_T, locy)
        if args.distillation_point_method == 1:
            loss_loc = 0.5 * torch.sum((rs_loss - rt_loss) ** 2)
        elif args.distillation_point_method == 2:
            loss_loc = - torch.log(torch.div(1.0,torch.abs(rs_loss - rt_loss) + 1.0))
        else:
            st_loss = - torch.log(torch.div(1.0,torch.abs(rs_loss - rt_loss) + 1.0))
            loss_loc = torch.div((rs_loss + loss_w + st_loss), 3.0)
        return loss_loc


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

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class DriftPointFNN(nn.Module):
    """
    normal feed-network
    """
    def __init__(self,Data_Vector_Length):
        super(DriftPointFNN, self).__init__()
        self.fc1 = nn.Linear(Data_Vector_Length, 800)
        self.fc2 = nn.Linear(800, 800)
        self.out = nn.Linear(800, 1)
        # self.w_h = nn.Parameter(torch.Tensor(1, 800))
        # self.b_h = nn.Parameter(torch.Tensor(1))
        # self.fc2 = nn.Linear(800, 1)

    def forward(self, x,loc_W):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        # out = F.linear(x, self.w_h, self.b_h)
        # loss_h = torch.div(torch.sum((x - loc_H) **2),x.shape[0])
        # - torch.log(torch.div(1.0, torch.abs(rs_loss - rt_loss) + 1.0))
        # x = self.fc2(x)
        # loss_w = 0.5 * torch.sum((self.fc2.weight - loc_W) **2)
        loss_w = torch.div(torch.sum((self.out.weight - loc_W) ** 2), self.out.weight.shape[1])
        return out,loss_w

class DriftTypeFNN(nn.Module):
    """
    normal feed-network
    """
    def __init__(self,Data_Vector_Length):
        super(DriftTypeFNN, self).__init__()
        if Data_Vector_Length > 50:
            self.hidden_size = Data_Vector_Length * 2
        else:
            self.hidden_size = Data_Vector_Length * 3
        self.fc1 = nn.Linear(Data_Vector_Length, self.hidden_size * 2)
        self.fc2 = nn.Linear( self.hidden_size * 2, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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




