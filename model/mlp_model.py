from torch import nn

class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self, n_input_dim=48, classification=False):
        super().__init__()

        self.classification = classification

        n_hidden1 = 256
        n_hidden2 = 1024
        n_hidden3 = 4096
        n_hidden4 = 1024
        n_hidden5 = 256
        n_output = 1

        self.layer_1 = nn.Linear(n_input_dim, n_hidden1)
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_3 = nn.Linear(n_hidden2, n_hidden3)
        self.layer_4 = nn.Linear(n_hidden3, n_hidden4)
        self.layer_5 = nn.Linear(n_hidden4, n_hidden5)
        self.layer_out = nn.Linear(n_hidden5, n_output)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        self.batchnorm3 = nn.BatchNorm1d(n_hidden3)
        self.batchnorm4 = nn.BatchNorm1d(n_hidden4)
        self.batchnorm5 = nn.BatchNorm1d(n_hidden5)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.relu(self.layer_5(x))
        x = self.batchnorm5(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        if self.classification:
            x = self.sigmoid(x)
        return x
