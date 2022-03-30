import torch
import torch.nn.functional as F

class QModel(torch.nn.Module):
    def __init__(self, num_layers=10, embedding_size=10):
        super(QModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=embedding_size)
        
        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(embedding_size),
                torch.nn.ReLU()
                ) for _ in range(num_layers)
        ])
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(81 * embedding_size, embedding_size),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.ReLU())
        self.output_layer = torch.nn.Linear(embedding_size, 1)

    def forward(self, state):
        x = self.embedding(state).permute(0, 3, 1, 2)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.linear_layer(x)
        return self.output_layer(x).squeeze()