**GAT-LSTM Model for Power Load Forecasting**

This repository contains the implementation of the GAT-LSTM model, a hybrid approach that combines Graph Attention Networks (GAT) and Long Short-Term Memory Networks (LSTM) for short-term forecasting of power load. The model leverages the spatio-temporal dependencies in energy systems, incorporating graph-structured data (e.g., power grid topology) and temporal sequences (e.g., historical energy consumption and weather data).

**Features**

i. Edge Attribute Conditioning: Transforms edge features to influence GAT attention mechanisms effectively.

ii. Graph Attention Network (GAT): Utilizes parallel GAT layers to capture spatial relationships and node-level interactions in a graph representing the power grid.

iii. Spatial and Temporal Fusion: Combines the graph-derived embeddings with the sequence (temporal) data before feeding to the sequence processor.

iv. LSTM: Processes the fused, temporally-aware embeddings to make the final hourly power load prediction.


**Data 	Source**

i. Electricity	(Load, PV, wind, etc.):	https://gitlab.com/dlr-ve/esy/open-brazil-energy-data/open-brazilian-energy-data

ii. Grid	(Line length, capacity, efficiency, etc.): https://gitlab.com/dlr-ve/esy/open-brazil-energy-data/open-brazilian-energy-data

iii. Weather	(Air temperature, pressure, rainfall, etc.): https://www.kaggle.com/datasets/gregoryoliveira/brazil-weather-information-by-inmet?resource=download

iv. Socio-economic	(State-wise GDP): https://www.ibge.gov.br/en/statistics/economic/national-accounts/16855-regional-accounts-of-brazil.html

v. Population	(State-wise population): https://www.ibge.gov.br/en/statistics/social/population/18448-estimates-of-resident-population-for-municipalities-and-federation-units.html?edicao=28688
