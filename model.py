import torch


import torch


class ModelLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, output_size, n_reps):
        super(ModelLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.n_reps = n_reps
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
            ) 
    
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x, xf):
        x_init = xf.unsqueeze(0)
        x_init = x_init.repeat(self.num_layers, 1, self.n_reps)
        output, h_n = self.rnn(x, x_init)
        output, seq_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        output = self.output_layer(output).squeeze()
        return output
    
    def predict(self, x, xf, is_state=False):
        if is_state:
            output, h_n = self.rnn(x.unsqueeze(0).unsqueeze(0), xf)
        else:
            x_init = xf.unsqueeze(0)
            x_init = x_init.repeat(self.num_layers, 1, self.n_reps)     
            output, h_n = self.rnn(x.unsqueeze(0).unsqueeze(0), x_init)
        output = self.output_layer(output).squeeze()
        return output, h_n



class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, output_size, n_reps):
        super(Model, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.n_reps = n_reps
        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
            ) 
    
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x, xf):
        x_init = xf.unsqueeze(0)
        x_init = x_init.repeat(self.num_layers, 1, self.n_reps)
        output, h_n = self.rnn(x, x_init)
        output, seq_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        output = self.output_layer(output).squeeze()
        return output
    
    def predict(self, x, xf, is_state=False):
        if is_state:
            output, h_n = self.rnn(x.unsqueeze(0).unsqueeze(0), xf)
        else:
            x_init = xf.unsqueeze(0)
            x_init = x_init.repeat(self.num_layers, 1, self.n_reps)     
            output, h_n = self.rnn(x.unsqueeze(0).unsqueeze(0), x_init)
        output = self.output_layer(output).squeeze()
        return output, h_n
