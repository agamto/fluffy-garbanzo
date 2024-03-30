import math
import torch
import torch.nn.functional as F
from baseModels import LSTMCell
import torch.nn as nn


class VanillaLSTM(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(VanillaLSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.lstm_cell = LSTMCell(input_size, hidden_size)

  def forward(self, inputs, hx=None):
    outputs = []
    if hx is None:
      hx = (torch.zeros(inputs.size(0), self.hidden_size),
            torch.zeros(inputs.size(0), self.hidden_size))
    for input in inputs.chunk(inputs.size(1), dim=1):
      hx = self.lstm_cell(input.squeeze(1), hx)
      outputs.append(hx[0].unsqueeze(1))
    outputs = torch.cat(outputs, dim=1)
    return outputs, hx

