# RATD
This repository contains the experiments in the work [Retrieval-Augmented Diffusion Models
for Time Series Forecasting] by Jingwei Liu, Ling Yang, Hongyan Li and Shenda Hong.

## Requirement

Please install the packages in requirements.txt

## Notice

This version is not the final version of our code, we will wpdate the full version ASAP.

## Preparation

### Download the elecricity dataset 
The data can e found at ./data/ts2vec

## Experiments 

### Retrieval
We use the TCN as our encoder, the code can be found at ./TCN-master. 
```shell
python retrieval.py --type encode
```
To save the references, you can run

```shell
python retrieval.py --type retrieval
```
### Training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity --nsample [number of samples]
```

## Acknowledgements

Codes are based on [CSDI](https://github.com/ermongroup/CSDI)



