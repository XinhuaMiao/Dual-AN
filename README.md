# Dual-AN
This is the offcial repository of paper "Dual-AN: A Hierarchical Framework Synergizes Frequency and Time Domain for Non-stationary Time Series Forecasting"



![image](https://github.com/XinhuaMiao/Dual-AN/blob/main/fig/Overview%20of%20Dual-AN.png)

# 1 Datasets

ETTh1, ETTh2, ETTm1, ETTm2, ExchangeRate, Weather, Electricity will be downloaded automatically.

# 2 Preparation

## 2.1 install requirements

1. to run our code, **Please make sure your python version is >=3.8.**
2. install all the requirements, we assume that you have installed torch according to your environment:
```
pip install -r ./requirements.txt
```
s

Please first source this init.sh script:

```
source ./init.sh 
```

or manually add this directory to your PATHONPATH environment variable:

```
export PYTHONPATH=./
```

# 3 Train and Evaluation

## 3.1 Dual-AN


Please change the settings in the following for what you need.
```python
# running Dual-AN using DLinear backbone with output length 96, 168, 336, 720 on dataset ETTh1 with input window 96, and hyperparameter k = 4
bach ./scripts/run_dualan.sh "DLinear" "DualAN"  "ETTh1" "96 168 336 720"  "cuda:0" 96  "{freq_topk:4}"
```

## 3.2 FAN

Please change the settings in the following for what you need.
```python
# running FAN using Dlinear backbone with output length 96, 168, 336, 720 on dataset ETTh1 with input window 96, and hyperparameter k = 4
bash ./scripts/run_dualan.sh "DLinear" "FAN" "ETTh1" "96 168 336 720"  "cuda:0" 96  "{freq_topk:4}"
```

## 3.3 Other Baselines
Please change the settings in the following for what you need.
```python
# running all other baselines (DLinear backbone) with output length 96, 168, 336, 720 on dataset ETTh1 ETTh2 with input window 96
bash ./scripts/run.sh "DLinear" "No RevIN DishTS SAN" "ETTh1 ETTh2" "96 168 336 720"  "cuda:0" 96
```

## 3.4 Wandb Tool
If you want to use the **wandb** tool, please change run_dualan.sh to run_dualan_wandb.sh:

```python
# running Dual-AN using Dlinear backbone with output length 96, 168, 336, 720 on dataset ETTh1 with input window 96, and hyperparameter k = 4
bash ./scripts/run_dualan_wandb.sh "DLinear" "DualAN" "ETTh1" "96 168 336 720"  "cuda:0" 96  "{freq_topk:4}"
```

or change run.sh to run_wandb.sh:

```python
# running Dual-AN using Dlinear backbone with output length 96, 168, 336, 720 on dataset ETTh1 ETTh2 with input window 96, and hyperparameter k = 4
bash ./scripts/run_wandb.sh "DLinear" "No RevIN DishTS SAN" "ETTh1 ETTh2" "96 168 336 720"  "cuda:0" 96
```

# 4 Others
For other parts that we follow the FAN method, such as data scaling, data split, z-score, etc., please refer to the code repository of [FAN](https://github.com/wayne155/FAN).
