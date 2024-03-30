import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
from utils.utilactivation import utilactivation

class MakeMLP(nn.Module):
    def __init__(self, args, layer_num, layer_name, input_num, hunit_num, output_num, active_fun, drop_ratio, ifbias,
                 iflastac=True):
        super(MakeMLP, self.__init__())
        layers = []
        self.args = args
        if ifbias:
            lastac = active_fun
            lastdrop = drop_ratio
        else:
            lastac = ''
            lastdrop = 0
        if layer_num > 1:
            self.addLayer(layers, input_num, hunit_num, ifbias, active_fun, drop_ratio)
            for i in range(layer_num - 2):
                self.addLayer(layers, hunit_num, hunit_num, ifbias, active_fun, drop_ratio)
            self.addLayer(layers, hunit_num, output_num, ifbias, lastac, lastdrop)
        else:
            self.addLayer(layers, input_num, output_num, ifbias, lastac, lastdrop)
        self.MLP = nn.Sequential(*layers)
        if layer_name == 'rel':
            self.MLP.apply(self.init_wieight_rel)
        elif layer_name == 'nei':
            self.MLP.apply(self.init_wieight_nei)
        elif layer_name == 'attR':
            self.MLP.apply(self.init_weights_ngate)

    def addLayer(self, layers, input_num, output_num, ifbias, active_fun, drop_ratio):
        layers.append(nn.Linear(input_num, output_num, bias=ifbias))
        Active_fun = nn.ReLU
        if active_fun == 'sig':
            Active_fun = nn.Sigmoid
            layers.append(Active_fun())
        elif active_fun == 'relu':
            Active_fun = nn.ReLU
        elif active_fun == 'tanh':
            Active_fun = nn.Tanh
        elif active_fun == 'lrelu':
            Active_fun = nn.LeakyReLU
        layers.append(Active_fun())
        if drop_ratio != 0:
            layers.append(nn.Dropout(drop_ratio))
        return layers

    def init_wieight_nei(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal(m.weight, gain=self.args.nei_std)
            if self.args.ifbias_nei:
                nn.init.constant(m.bias, 0)

    def init_weights_rel(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=self.args.rela_std)
            if self.args.ifbias_rel:
                nn.init.constant(m.bias, 0)

    def init_weights_ngate(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.005)
            if self.args.ifbias_gate:
                nn.init.constant(m.bias, 0)


class LSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(LSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
    self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
    self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
    self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input, hx):
    hx, cx = hx

    gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
    ingate, forgotgate, cellgate, outgate_ = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgotgate = torch.sigmoid(forgotgate)
    cellgate = torch.tanh(cellgate)
    outgate_ = torch.sigmoid(outgate_)

    cy = forgotgate * cx + ingate * cellgate
    hy = outgate_ * torch.tanh(cy)

    return hy, cy
