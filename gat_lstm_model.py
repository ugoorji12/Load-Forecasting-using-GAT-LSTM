# gat_lstm_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT_LSTM(nn.Module):
    def __init__(self, node_feature_dim, sequence_feature_dim, gat_out_channels, gat_heads, lstm_hidden_dim, lstm_layers, edge_dim):
        super(GAT_LSTM, self).__init__()
        
        # Edge attribute conditioning
        self.edge_attr_transform = nn.Linear(edge_dim, gat_out_channels)

        # Parallel GAT layers
        self.gat_1hop_1 = GATConv(node_feature_dim, gat_out_channels, heads=gat_heads, concat=False, edge_dim=gat_out_channels)
        self.gat_1hop_2 = GATConv(node_feature_dim, gat_out_channels, heads=gat_heads, concat=False, edge_dim=gat_out_channels)

        self.gat_dropout = nn.Dropout(0.2)

        # The input to LSTM is the sequence feature dimension + the output of the GAT layers
        combined_input_dim = sequence_feature_dim + 2 * gat_out_channels  
        self.lstm = nn.LSTM(combined_input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)

        # Fully connected layer for final prediction
        self.fc = nn.Linear(lstm_hidden_dim, 1)  # Output layer to predict target consumption

    def forward(self, sequences, edge_index, edge_attr, node_features, node_indices):
        # Edge attribute conditioning
        transformed_edge_attr = self.edge_attr_transform(edge_attr)

        # GAT layer 1-hop_1 neighbors
        gat_1hop_out_1 = self.gat_1hop_1(node_features, edge_index, transformed_edge_attr)
        gat_1hop_out_1 = self.gat_dropout(gat_1hop_out_1)

        # GAT layer 1-hop_2 neighbors
        gat_1hop_out_2 = self.gat_1hop_2(node_features, edge_index, transformed_edge_attr)
        gat_1hop_out_2 = self.gat_dropout(gat_1hop_out_2)

        # Gather GAT output for each sequence's node index (both 1-hop and 2-hop)
        gat_1hop_out_1 = gat_1hop_out_1[node_indices]
        gat_1hop_out_2 = gat_1hop_out_2[node_indices]

        # Concatenate the GAT outputs (1-hop_1 and 1-hop_2) along the feature dimension
        gat_combined_out = torch.cat((gat_1hop_out_1, gat_1hop_out_2), dim=-1)  # [batch_size, 2 * gat_out_channels]

        # Expand the GAT output to match the sequence length and concatenate with the sequence data
        gat_combined_out = gat_combined_out.unsqueeze(1).repeat(1, sequences.size(1), 1)  # Repeat for each time step
        combined_input = torch.cat((sequences, gat_combined_out), dim=-1)  # [batch_size, seq_len, seq_feat_dim + 2 * gat_out_channels]

        # LSTM layer processes the combined sequence and GAT data
        lstm_out, _ = self.lstm(combined_input)
        lstm_out = self.lstm_dropout(lstm_out)

        # Take the last output of LSTM
        lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]

        # Fully connected layer to predict the next hour's consumption
        out = self.fc(lstm_out)
        
        return out
