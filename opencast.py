import torch
import torch.nn as nn
from torch_geometric.nn import FAConv
from oneclass import one_class_loss, one_class_masking, One_Class_GNN_prediction
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn.models.autoencoder import GAE

class OPENCAST(nn.Module):
    def __init__(self, input_len, hidden_lens, num_layers, eps):
        super(OPENCAST, self).__init__()
        self.num_layers = num_layers
        
        self.layer1 = FAConv(channels=input_len, eps=eps, dropout=0.1, cached=True)
        
        if num_layers == 2:
            self.layer2 = FAConv(channels=input_len, eps=eps, dropout=0.1, cached=True)
        
        self.layer3 = nn.Linear(input_len, hidden_lens[0])
        self.layer4 = nn.Linear(hidden_lens[0], hidden_lens[1])

    def forward(self, x, edge_index):
        h1 = nn.LeakyReLU(0.2)(self.layer1(x=x, x_0=x, edge_index=edge_index)) 
        
        if self.num_layers == 2:
            h1 = nn.LeakyReLU(0.2)(self.layer2(x=h1, x_0=x, edge_index=edge_index)) 

        h2 = nn.LeakyReLU(0.2)(self.layer3(h1))
        h3 = nn.Tanh()(self.layer4(h2))
        return h3

def train_OPENCAST(g, hidden1, hidden2, lr, r, multi_task, epochs, alpha, eps, num_layers):
    device = torch.device('cuda:0')
    loss_ocl = torch.Tensor([0]).to(device)
    center = torch.Tensor([0] * hidden2).to(device)
    radius = torch.Tensor([r]).to(device)

    mask, t_mask, mask_unsup, t_mask_unsup = one_class_masking(g, False)
    G = from_networkx(g).to(device)
    g_unsup = g.subgraph(t_mask_unsup)
    G_unsup = from_networkx(g_unsup).to(device)
    model_ocl = OPENCAST(len(G.features[0]), [hidden1, hidden2], num_layers, eps)
    model = GAE(model_ocl).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs+1):
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        learned_representations = model.encode(G.features.float(), G.edge_index)

        if epoch < multi_task:
            recon_loss = model.recon_loss(learned_representations, G.edge_index)
            loss = recon_loss
        else:
            loss_ocl = one_class_loss(center, radius, learned_representations, mask)

            recon_loss = model.recon_loss(learned_representations[mask_unsup], G_unsup.edge_index)

            loss = (loss_ocl*alpha) + ((1-alpha) * recon_loss)
           
        # Compute gradients
        loss.backward()

        # Tune parameters
        optimizer.step()

    return One_Class_GNN_prediction(center, radius, learned_representations, g, 'test', True)['macro avg']['f1-score']
