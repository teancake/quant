from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def get_outputs(self, seq_inputs, cat_inputs):
        pass

    @abstractmethod
    def get_loss(self, targets, seq_inputs, cat_inputs):
        pass



class EmbeddingLayer(nn.Module):
    def __init__(self, cat_input_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding_vocab_size = 1000
        self.embedding_dim = 16
        self.output_size = cat_input_size * self.embedding_dim
        self.embedding_layer = self.init_embedding_layer(cat_input_size)
        # self.fc = nn.Linear(cat_input_size * self.embedding_dim, output_size)

    def init_embedding_layer(self, cat_input_size):
        embedding_layer = []
        for i in range(cat_input_size):
            embedding = nn.Embedding(self.embedding_vocab_size, self.embedding_dim)
            embedding_layer.append(embedding)
        return embedding_layer

    def forward(self, cat_inputs):
        vecs = []
        for i, emb in enumerate(self.embedding_layer):
            # print("cat_inputs {} is {}".format(i, cat_inputs[:, :, i]))
            vec = emb(cat_inputs[:, :, i])
            # print("vec shape {} content {}".format(vec.shape, vec))
            vecs.append(vec)
        output = torch.cat(vecs, -1)
        # print("after concat all embeddings, output shape {} content {}".format(output.shape, output))
        # output = self.fc(output)
        return output

    # def get_loss(self, targets, seq_inputs, cat_inputs):
    #     outputs = self.get_outputs(seq_inputs, cat_inputs)
    #     # print("model outputs shape {}".format(outputs.shape))
    #     # print("inputs {}, targets {}, outputs {}".format(inputs, targets, outputs))
    #     # loss = loss_fn(outputs[:,-1], targets[:,-1])
    #     loss = torch.nn.MSELoss()(outputs, targets)
    #     return loss

