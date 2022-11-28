# Carmen
AAAI-23 paper: Context-aware Safe Medication Recommendations with Molecular Graph and DDI Graph Embedding
This is an implementation of our model Carmen and the baselines in the paper. 
<hr>

## Requirements
```python
torch == 1.8.0+cu111
torch-geometric == 1.0.3
torch-scatter == 2.0.9
torch-sparse == 0.6.12
```

## Process Data
```python
\Carmen\data
```
The processed data is in the path, you can also process data with:
### MIMIC-III
```python
python processing.py
```
### MIMIC-IV
```python
python processing_4.py
```

## Train Model
```python
\Carmen\src
```
```python
python main_train.py --cuda 2 --datadir ../data/ --MIMIC 3 --model_name Model_mimic3_DDIenc --encoder main --seed 312 --ddi_encoding --gnn_type gat --num_layer 10 --p_or_m minus
```

## Test Model
```python
python main_train.py --cuda 2 --datadir ../data/ --MIMIC 3 --model_name Model_mimic3_DDIenc --encoder main --seed 312 --ddi_encoding --gnn_type gat --num_layer 10 --p_or_m minus --Test --resume_path saved/Model_mimic3_DDIenc/best.model
```
