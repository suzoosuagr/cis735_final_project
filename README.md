# CIS 735 Final Project

# Member

- **Jiyang Wang**: implementation about our proposed method
  - jwang127@syr.edu
- **Weiheng Chai** Chai: Implementation about baseline and ensemble method
  - wchai01@syr.edu
- **Kun Wu**: Implementation about baseline and data splits
  - kwu102@syr.edu

# Usage

## docker
```bash
# build image
cd ./Docker
docker build -t pytorch1.5.0:cis735 --rm .
```

## prepare data
please go to https://www.kaggle.com/c/state-farm-distracted-driver-detection and download the dataset. And unzip it to the `../data/statefarm/` folder. After unzip the folder structure should be like:

```
..
|── data
│   └── statefarm             # The StateFarm Dataset
│       └── imgs
│           ├── test          
│           └── train
│               ├── c0
│               ├── c1
│               ├── c2
│               ├── c3
│               ├── c4
│               ├── c5
│               ├── c6
│               ├── c7
│               ├── c8
│               └── c9
|
|
└── cis735_final_project(*)   # current dir
    ├── Dataset               # Defines dataset
    │   ├── instruction
    │   └── __pycache__
    ├── Docker                # Dockerfile
    ├── Experiments           
    │   └── Config            # Config files
    │       └── __pycache__
    ├── Log                   
    │   ├── ISSUE01_EXP1      # << Logs our method
    │   ├── ISSUE01_EXP2
    │   └── ISSUE01_EXP3
    ├── Model             
    │   └── __pycache__
    ├── __pycache__
    ├── Recipes
    ├── temp
    ├── TFrecords            # Tensorboard Records
    │   ├── ISSUE01_EXP1
    │   │   └── 2021_05_11_21_18_31
    │   ├── ISSUE01_EXP2
    │   │   └── 2021_05_12_14_40_23
    │   └── ISSUE01_EXP3
    │       └── 2021_05_13_21_17_10
    └── Tools
        └── __pycache__
```

## Python packages:
- please see : [requirements.txt](./Docker/requirements.txt)


## run our model

1. Train:
    ```
    python main_entry.py -m train -l train.log
    ```

2. Evaluate validation accuracy
   ```
   python main_entry.py -m valid -l valid.log
   ``` 

3. Predicting test split and generate the submission csv file
   ```
   python main_entry.py -m submit -l submit.log
   ```

## run baseline
Please check our baseline model repository at : https://github.com/wowowoxuan/statefarm_task
1. Train
    change the \<MODEL\> to run different experiments
    ```
    python train<MODEL>.py
    ```
