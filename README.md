**GAT-LSTM Model for Power Load Forecasting**

This repository contains the implementation of the GAT-LSTM model, a hybrid approach that combines Graph Attention Networks (GAT) and Long Short-Term Memory Networks (LSTM) for short-term forecasting of power load and renewable energy supply. The model leverages the spatio-temporal dependencies in energy systems, incorporating graph-structured data (e.g., power grid topology) and temporal sequences (e.g., historical energy consumption and weather data).

**Features**

i. Edge Attribute Conditioning: Transforms edge features to influence GAT attention mechanisms effectively.

ii. Graph Attention Network (GAT): Utilizes parallel GAT layers to capture spatial relationships and node-level interactions in a graph representing the power grid.

iii. Spatial and Temporal Fusion: Combines the graph-derived embeddings with the sequence (temporal) data before feeding to the sequence processor.

iv. LSTM: Processes the fused, temporally-aware embeddings to make the final hourly power load prediction.
