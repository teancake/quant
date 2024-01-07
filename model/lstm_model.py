from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from base_model import BaseModel, EmbeddingLayer
from deepar import DeepAR



class LstmEmb(BaseModel):
    def __init__(self, seq_input_size, cat_input_size, output_size=1, lstm_hidden_size=512,
                 lstm_num_layers=1, lstm_dropout=0.0):
        super(LstmEmb, self).__init__()
        self.lstm = LSTM(input_size=seq_input_size, output_size=lstm_hidden_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_num_layers, dropout=lstm_dropout, use_attention=False)

        self.embedding = EmbeddingLayer(cat_input_size)

         # DNN 层
        dnn_input_size = self.lstm.output_size + self.embedding.output_size

        self.linears = nn.Sequential(nn.Linear(dnn_input_size, 1024),
                                     nn.LeakyReLU(), nn.BatchNorm1d(1),
                                     nn.Linear(1024, 256), nn.LeakyReLU(),
                                     nn.BatchNorm1d(1), nn.Linear(256, 64),
                                     nn.LeakyReLU(), nn.BatchNorm1d(1),
                                     nn.Linear(64, 16), nn.LeakyReLU(),
                                     nn.BatchNorm1d(1), nn.Dropout(0.1))

        self.fc = nn.Linear(16, output_size)

    def forward(self, seq_inputs, cat_inputs):
        seq_outputs, _ = self.lstm(seq_inputs)
        emb_outputs = self.embedding(cat_inputs)
        # print("seq_outputs shape {}, emb_outputs shape {}, seq_outputs[:, [-1], :] shape {}".format(seq_outputs.shape, emb_outputs.shape, seq_outputs[:, [-1], :].shape))
        cat_outputs = torch.cat([seq_outputs[:, [-1], :], emb_outputs], -1)
        # print("cat_outputs shape {}".format(cat_outputs.shape))
        outputs = self.linears(cat_outputs)
        outputs = self.fc(outputs)
        # print("after fc outputs shape {}".format(outputs.shape))
        return outputs

    def get_outputs(self, seq_inputs, cat_inputs):
        return self.forward(seq_inputs, cat_inputs)

    def get_loss(self, targets, seq_inputs, cat_inputs):
        outputs = self.get_outputs(seq_inputs, cat_inputs)
        # print("model outputs shape {}".format(outputs.shape))
        # print("inputs {}, targets {}, outputs {}".format(inputs, targets, outputs))
        # loss = loss_fn(outputs[:,-1], targets[:,-1])
        loss = torch.nn.MSELoss()(outputs, targets[:, [-1]])
        return loss


class LSTM(BaseModel):
    def __init__(self, input_size, output_size=1, hidden_size=512, num_layers=1, dropout=0.0, use_attention=False):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.bn_in = nn.BatchNorm1d(sequence_length, affine=False)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.use_attention = use_attention
        self.output_size = output_size

    def forward(self, x, hidden=None):
        # print("x size {}".format(x.shape))
        batch_size = x.size(0)
        # print("x before bn {}".format(x))
        # print("x shape {}, before batch norm {}".format(x.shape, x))
        # x = self.bn_in(x)
        # print("x shape {}, after batch norm {}".format(x.shape, x))
        # print("x after bn {}".format(x))

        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)


        # print("lstm_out size {}, hidden 0 size {}, hidden 1 size {}".format(lstm_out.shape, hidden[0].shape, hidden[1].shape))

        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # lstm_out = self.bn_out(lstm_out)
        # print("lstm_out before drop out {}".format(lstm_out.shape))
        if self.use_attention:
            # print("lstm_out before attention {}".format(lstm_out.shape))
            sequence_length = lstm_out.shape[1]
            attn_mask = torch.tril(torch.ones((sequence_length, sequence_length))) == 0
            attn_out, attn_weight = self.mha(lstm_out, lstm_out, lstm_out,  attn_mask=attn_mask)
            # print("attention weight {}".format(attn_weight))
            out = attn_out
        else:
            out = lstm_out
        # print("out before fc {}".format(out.shape))
        out = self.fc(out)
        # out = self.head(out)
        # print("out after fc {}".format(out.shape))
        return out, hidden

    def init_hidden(self, batch_size, method="zero"):
        weight = next(self.parameters()).data
        if method == "zero":
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).normal_(0, 0.01).to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_size).normal_(0, 0.01).to(device))

        return hidden

    def get_outputs(self, seq_inputs, cat_inputs=None):
        outputs, _ = self.forward(seq_inputs)
        return outputs

    def get_loss(self, targets, seq_inputs, cat_inputs=None):
        outputs = self.get_outputs(seq_inputs)
        # print("model outputs shape {}".format(outputs.shape))
        # print("inputs {}, targets {}, outputs {}".format(inputs, targets, outputs))
        # loss = loss_fn(outputs[:,-1], targets[:,-1])
        loss = torch.nn.MSELoss()(outputs, targets)
        return loss





class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AeLstm(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(AeLstm, self).__init__()

        self.noise = AddGaussianNoise(std=0.2)

        ae_hidden_size = hidden_size
        lstm_hidden_size = hidden_size

        self.ae_encoder = nn.Linear(input_size, ae_hidden_size)
        self.ae_act = nn.ReLU()
        self.ae_decoder = nn.Linear(ae_hidden_size, input_size)
        self.rnn = nn.LSTM(input_size + ae_hidden_size, lstm_hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size + input_size, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, 1),
        )

    def forward(self, _x, _y=None):
        if self.training:
            h = self.noise(_x)
        else:
            h = _x

        ae_h1 = self.ae_act(self.ae_encoder(h))
        ae_h2 = self.ae_decoder(ae_h1)
        ae_loss = nn.MSELoss()(ae_h2, h)
        # print("ae_h1 {}, ae_h2 {}, ae_loss {}, {}".format(ae_h1.shape, ae_h2.shape, ae_loss.shape, ae_loss))
        # print("_x shapes {}, ae_h1 shapes {}".format(_x.shape, ae_h1.shape))
        h = torch.cat([_x, ae_h1], dim=-1)
        # print("h shapes {}".format(h.shape))

        h, _ = self.rnn(h)
        h = self.dropout(h)
        h = torch.cat([_x, h], dim=-1)
        y = self.head(h)
        # y = y.squeeze(1)
        # print("y shape {}, _y shape {}".format(y.shape, _y.shape))

        if _y is None:
            return y, None

        loss = nn.MSELoss()(y, _y.squeeze(1))
        # print("loss shape {}, ae_loss shape {}".format(loss.shape, ae_loss.shape))

        loss = loss + ae_loss

        return y, loss

    def get_outputs(self, seq_inputs, cat_inputs=None):
        outputs, loss = self.forward(seq_inputs)
        return outputs

    def get_loss(self, targets, seq_inputs, cat_inputs=None):
        outputs, loss = self.forward(seq_inputs, targets)
        return loss


def get_model(config) -> BaseModel:
    model_name = config.get("model_name")
    if model_name == "lstm":
        input_size = config.get("input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        dropout = config.get("dropout")
        model = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_name == "ae_lstm":
        input_size = config.get("input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        dropout = config.get("dropout")
        model = AeLstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_name == "lstm_attn":
        input_size = config.get("input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        dropout = config.get("dropout")
        model = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers,
                     dropout=dropout, use_attention=True)
    elif model_name == "lstm_emb":
        input_size = config.get("input_size")
        cat_input_size = config.get("cat_input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        dropout = config.get("dropout")
        model = LstmEmb(seq_input_size=input_size, cat_input_size=cat_input_size, output_size=output_size,
                        lstm_hidden_size=hidden_size, lstm_num_layers=num_layers, lstm_dropout=dropout)
    elif model_name == "deepar":
        input_size = config.get("input_size")
        cat_input_size = config.get("cat_input_size")
        output_size = config.get("output_size")
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        dropout = config.get("dropout")
        model = DeepAR(seq_input_size=input_size, cat_input_size=cat_input_size, output_size=output_size,
                        lstm_hidden_size=hidden_size, lstm_num_layers=num_layers, lstm_dropout=dropout)
    else:
        print("model {} not found.".format(model_name))
        model = None
    model.to(device)
    return model

if __name__ == '__main__':
    # lstm = torch.nn.LSTM(input_size=3, hidden_size=1)
    output_size = 2
    lstm = torch.nn.LSTM(3,output_size)
    print(lstm)
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, output_size),
              torch.randn(1, 1, output_size))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)


    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, output_size), torch.randn(1, 1, output_size))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)
