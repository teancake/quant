import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import torch.nn as nn

import uuid
import sys
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_dim=512, n_layers=1, drop_prob=0.0, sequence_length=1):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bn_in = nn.BatchNorm1d(sequence_length)
        self.bn_out = nn.BatchNorm1d(hidden_dim)


        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # print("x size {}".format(x.shape))
        batch_size = x.size(0)
        # print("x before bn {}".format(x))
        x = self.bn_in(x)
        # print("x after bn {}".format(x))
        lstm_out, hidden = self.lstm(x, hidden)
        # print("lstm_out size {}, hidden 0 size {}, hidden 1 size {}".format(lstm_out.shape, hidden[0].shape, hidden[1].shape))

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        lstm_out = self.bn_out(lstm_out)
        # print("lstm_out before drop out {}".format(lstm_out.shape))
        out = self.dropout(lstm_out)
        # print("out before fc {}".format(out.shape))
        out = self.fc(out)
        # print("out after fc {}".format(out.shape))

        out = out.view(batch_size, -1)
        out = out[:, -1]
        # print("out after -1 {}".format(out.shape))
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



def load_data_from_file(file_path):
    file = open(file_path, "rb")
    data = pickle.load(file)
    file.close()
    return data


def _get_sequential_data_nparray(symbol_df, sequence_length, number_of_sequences):
    list_x = []
    list_y = []
    list_ext = []
    # print("symbol {}, number of dates {}, min dates {}".format(symbol, symbol_df.shape[0], symbol_df["日期"].min()))
    for i in range(symbol_df.shape[0] - sequence_length + 1):
        date_df = symbol_df[i:i + sequence_length]
        # print("i {}, len {}, min date {}".format(i, date_df.shape, date_df["日期"].min()))
        x = date_df.loc[:, "ma_5":"turnover_rate"].fillna(0).values.astype(float)
        y = date_df.loc[:, "label"].values.astype(float)[0]
        ext = date_df.loc[:, ["日期", "代码", "label"]]
        ext = ext.iloc[[0]]

        list_x.append(x)
        list_y.append(y)
        list_ext.append(ext)
        if number_of_sequences is not None and i >= number_of_sequences - 1:
            break
    return list_x, list_y, list_ext


def get_sequential_data(df, sequence_length=1, number_of_sequences=None):
    POOL_SIZE = 16
    pool = ProcessPoolExecutor(max_workers=POOL_SIZE)
    data_x = []
    data_y = []
    data_ext = []
    futures = []
    for symbol in tqdm(df["代码"].unique()):
        symbol_df = df[df["代码"] == symbol].sort_values(by=["日期"], ascending=False)
        future = pool.submit(_get_sequential_data_nparray, symbol_df, sequence_length, number_of_sequences)
        futures.append(future)
        if len(futures) % POOL_SIZE == POOL_SIZE - 1:
            # the main purpose of waiting is to make tqdm correct.
            wait(futures)
        # if symbol > "000070":
        #     break

    wait(futures)

    for future in futures:
        list_x, list_y, list_ext = future.result()
        data_x.extend(list_x)
        data_y.extend(list_y)
        data_ext.extend(list_ext)

    pool.shutdown(wait=True)

    data_x = np.stack(data_x, axis=0)
    data_y = np.stack(data_y, axis=0)
    data_ext = pd.concat(data_ext, ignore_index=True, sort=False)

    print("sequential data x shape {}, sequential data y shape {}, ext data shape {}".format(data_x.shape, data_y.shape, data_ext.shape))
    return data_x, data_y, data_ext




def train():
    sequence_length = 15
    batch_size = 1001
    learning_rate = 1e-2
    epoch_num = 5
    dataset = load_data_from_file("/mnt/data/quant_reg_data.pkl")
    train_data, test_data, pred_data = dataset

    train_data = train_data[train_data["日期"] > train_data["日期"].max() - timedelta(days=365)]
    print("train_data min date {}, max date {}".format(train_data["日期"].min(), train_data["日期"].max()))
    print("test_data min date {}, max date {}".format(test_data["日期"].min(), test_data["日期"].max()))


    print("convert training data into sequences.")
    train_data_x, train_data_y, _ = get_sequential_data(train_data, sequence_length)
    print("convert testing data into sequences.")
    test_data_x, test_data_y, test_data_ext = get_sequential_data(test_data, sequence_length)
    print("convert prediction data into sequences")
    pred_data_x, _, pred_data_ext = get_sequential_data(pred_data, sequence_length, number_of_sequences=1)

    print("train x shape {}, train y shape {}, train y mean {}, variance {}".format(train_data_x.shape, train_data_y.shape, np.mean(train_data_y), np.var(train_data_y)))
    print("test x shape {}, test y shape {}, test y mean {}, variance {}".format(test_data_x.shape, test_data_y.shape, np.mean(test_data_y), np.var(test_data_y)))
    print("pred x shape {}, append x shape {}".format(pred_data_x.shape, pred_data_ext.shape))

    print("data loaded")
    train_data_x = torch.tensor(train_data_x, dtype=torch.float32)
    train_data_y = torch.tensor(train_data_y, dtype=torch.float32)
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y, dtype=torch.float32)
    pred_data_x = torch.tensor(pred_data_x, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_x, train_data_y)
    val_dataset = TensorDataset(test_data_x, test_data_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    model = LSTM(input_size=train_data_x.shape[-1], n_layers=5, drop_prob=0.8, sequence_length=sequence_length)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("model created {}".format(model))

    # Run the training loop
    h = model.init_hidden(batch_size)

    for epoch in range(0, epoch_num):
        print("Starting epoch {}".format(epoch + 1))

        running_loss = 0.0
        for batch_num, data in enumerate(train_loader):

            model.train(True)
            optimizer.zero_grad()
            inputs, targets = data
            # print("input shape {}, target shape {}, h0 shape {}, h1 shape {}".format(inputs.shape, targets.shape, h[0].shape, h[1].shape))
            h = tuple([each.data for each in h])
            outputs, _ = model(inputs, h)
            # print("model outputs shape {}".format(outputs.shape))

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_num % 50 == 0:
                avg_loss = running_loss / (batch_num + 1)
                print("Loss after batch {}: {}".format(batch_num, avg_loss))
        avg_loss = running_loss / (batch_num + 1)
        print("Training process in epoch {} has finished. Evaluation started.".format(epoch + 1))

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            val_h = model.init_hidden(batch_size)
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                val_h = tuple([each.data for each in val_h])
                voutputs, _ = model(vinputs, val_h)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print("epoch {} LOSS train {} val {}".format(epoch + 1, avg_loss, avg_vloss))

    run_id = uuid.uuid1().hex
    model_name = "lstm"
    print("saving validation results")
    pred, _ = model(test_data_x, model.init_hidden(test_data_x.shape[0]))
    test_data_ext["score"] = pred.squeeze().detach().numpy()

    print("now make predictions")
    pred_h = model.init_hidden(pred_data_x.shape[0])
    pred, _ = model(pred_data_x, pred_h)
    print("pred shape {}".format(pred.shape))
    pred_data_ext["score"] = pred.squeeze().detach().numpy()
    print("now find the ups")
    pred_data_ext.sort_values(by=["score"], inplace=True, ascending=False)
    print(pred_data_ext.head(20).to_string())



if __name__ == "__main__":
    train()


