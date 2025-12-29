import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool

class GATModel(torch.nn.Module):
    """
    Graph Attention Network (GAT) for binary classification of molecular graphs.
    
    Architecture:
    - 3 GAT Convolutional layers (configurable)
    - Global Mean Pooling
    - 2-layer MLP classifier
    
    Args:
        num_node_features (int): Number of input features per node.
        num_edge_features (int): Number of input features per edge.
        hidden_dim (int): Hidden dimension size.
        heads (int): Number of attention heads.
        num_layers (int): Number of GAT layers.
        dropout (float): Dropout probability.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, heads=4, num_layers=3, dropout=0.2):
        super(GATModel, self).__init__()
        self.dropout_ratio = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        # Note: GATConv can handle edge attributes if configured, but standard GAT doesn't always use them effectively without modification.
        # PyG's GATConv supports edge_dim.
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=heads, edge_dim=num_edge_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=num_edge_features))
            
        # Last conv layer (project to hidden_dim without heads concatenation or keep heads and project later)
        # Usually we keep heads and concat.
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=num_edge_features))
        
        # Pooling -> MLP
        # Input to MLP is hidden_dim * heads
        self.lin1 = Linear(hidden_dim * heads, hidden_dim)
        self.lin2 = Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
           data (torch_geometric.data.Data): Input graph data batch.

        Returns:
           torch.Tensor: Logits (unnormalized scores) for binary classification.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # GAT Layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            
        # Global Pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim * heads]
        
        # MLP Classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        
        return x
