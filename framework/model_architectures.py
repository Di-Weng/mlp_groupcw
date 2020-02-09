import torch
import torch.nn as nn


class LSTMBlock(nn.Module):

    def __init__(self, input_dim, batch_size, hidden_dim, output_dim=4, num_layers=1):
        super(LSTMBlock, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.input_shape = (self.batch_size, 900, self.input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()

        print("BLSTM block using input shape", self.input_shape)

        out = torch.zeros(self.input_shape)

        self.layer_dict['blstm'] = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                           num_layers=self.num_layers, batch_first=True, dropout=0.5,
                                           bidirectional=True)

        out,_ = self.layer_dict['blstm'].forward(out)
        out = out[:, -1, :]
        print(out.shape)
        # print(type(out))
        # print(type(out[0]))
        # print(out[0].shape)
        # print(type(out[1]))
        # print(out[1].shape)
        self.layer_dict['linear'] = nn.Linear(self.hidden_dim * 2, self.output_dim)  # *2 for Bidirectional


        out = self.layer_dict['linear'].forward(out)

        print("Block is built, output volume is", out.shape)

        return out

    def forward(self, x):
        out = x
        out = self.layer_dict['blstm'].forward(out)
        out = self.layer_dict['linear'].forward(out)

        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

