from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
class SLSTMUnit(nn.Module):
    def __init__(self,u,options):
        super(SLSTMUnit, self).__init__()
        self.wis = nn.Linear(in_features= options.vols, out_features=  options.vols, bias = False)
        self.wix = nn.Linear(in_features= options.num_feature , out_features= options.vols , bias = False)
        self.wfs = nn.Linear(in_features= options.vols, out_features= options.vols, bias = False)
        self.wfx = nn.Linear(in_features= options.num_feature, out_features= options.vols, bias = False)
        self.we  = nn.Linear(in_features= options.vols, out_features= options.vols, bias = False)
        self.wd  = nn.Linear(in_features= options.num_feature, out_features= options.vols, bias = False)
        self.sigmoid = nn.functional.sigmoid()

    def forward(self,s, c, x, flag):
         i = self.sigmoid(self.wis(s)+self.wix(x))
         f = self.sigmoid(self.wfs(s)+self.wfx(x))
         c_hat = self.we(s)+self.wd(x)
         cc = f * c + i * c_hat
         s = torch.diag(torch.tanh(cc + u)+torch.tanh(cc-u))
         if flag:
             x = self.wd(s)
             return s,cc,x
         else:
             return s, cc


class adaptiveSLSTMUnit(nn.Module):
    def __init__(self, u, options):
        super(adaptiveSLSTMUnit, self).__init__()
        self.we  = nn.Linear(in_features= options.vols, out_features= options.vols, bias = False)
        self.wd  = nn.Linear(in_features= options.num_feature, out_features= options.vols, bias = False)

    def forward(self,s, c, x):

         c_hat = self.we(s)+self.wd(x)
         cc = f * c + i * c_hat
         s = torch.sign(cc)*max(torch.abs(cc)-self.lam*self.tau,0)
         return s, cc


class LSTM(nn.Module):
    def __init__(self,u):
        super(LSTM, self).__init__()
        self.SLSTMUnit = SLSTMUnit(u,options)
        self.adaptiveSLSTMUnit = adaptiveSLSTMUnit(options)
        self._init_weights()


    def _init_weights(self):
        class_name = self.SLSTMUnit.__class__.__name__
        if class_name.find('w') != -1:
            self.SLSTMUnit.weight.data.normal_(0, 0.02)

    def forward(self, layers,x,SS,CC):
        SS=[]
        CC=[]
        s = SS['0']
        c = CC['0']
        for i in range(1,layers):

            if i==0:
                s,c = self.adaptiveSLSTMUnit.forward(s,c,x)
                SS[str(i)]=s
                CC[str(i)]=c
            if i==(layers-1):
                flag = True
            if flag:
                s,c,xx = self.SLSTMUnit.forward(s, c, x, flag)
                SS[str(i)] = s
                CC[str(i)] = c

                return SS, CC, xx
            else:
                s,c = self.SLSTMUnit.forward(s, c, x, flag)
                SS[str(i)] = s
                CC[str(i)] = c