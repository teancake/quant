import torch

from mlp_model import MLP
import pickle
from torch.utils.data import Dataset, DataLoader

# from quant_data_util import load_data_from_db

class DataFrameDataset(Dataset):
    def __init__(self, df_x, df_y):
        self.x = torch.tensor(df_x, dtype=torch.float32)
        self.y = torch.tensor(df_y, dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def load_data(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def train():
    random_seed = 10
    data_file_name = "quant_reg_data.pkl"
    use_offline_data = True

    if use_offline_data:
        dataset = load_data(data_file_name)
#    else:
#        dataset = load_data_from_db()
#        pickle.dump(dataset, open(data_file_name, 'wb'))
#        print("data saved to {}".format(data_file_name))
    train_x, train_y, test_x, test_y, pred_x, append_x = dataset

    print("data loaded")
    print("train x shape {}, train y shape {}, train y mean {}, variance {}".format(train_x.shape, train_y.shape, train_y.mean(), train_y.var()))
    print("test x shape {}, test y shape {}, test y mean {}, variance {}".format(test_x.shape, test_y.shape, train_y.mean(), test_y.var()))
    print("pred x shape {}, append x shape {}".format(pred_x.shape, append_x.shape))

    train_dataset, val_dataset = DataFrameDataset(train_x, train_y), DataFrameDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=1)

    # train mlp
    model = MLP(n_input_dim=train_x.shape[1], classification=False)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("model created {}".format(model))

    # Run the training loop
    for epoch in range(0, 5):
        print("Starting epoch {}".format(epoch + 1))

        running_loss = 0.0
        for batch_num, data in enumerate(train_loader):

            model.train(True)
            optimizer.zero_grad()
            inputs, targets = data
            outputs = model(inputs)
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
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print("epoch {} LOSS train {} val {}".format(epoch + 1, avg_loss, avg_vloss))

    print("now make predictions")
    pred = model(torch.tensor(pred_x, dtype=torch.float32))
    append_x["score"] = pred.detach().numpy()
    print("now find the ups")
    append_x.sort_values(by=["score"], inplace=True)
    print(append_x.tail(20).to_string())


if __name__ == '__main__':
    train()
