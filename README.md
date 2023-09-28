# MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification (NeurIPS 2022)

## Setup
Step 1: Get repository 
```
https://github.com/RuiyuM/MVC.git
cd MVC
```
## Dataset Preparation make sure the dataset name is: modelnet40v2png_ori4
Step 1: Download the ModelNet40 dataset (20 view setting) and extract it to the current folder:
```
wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
tar -xvf modelnet40v2png_ori4.tar
```
Step 2: Place data.zip in this repository 
Step 3: Unzip data.zip 
```
unzip data.zip
```
# feature aggregation QUERIES_STRATEGY = ['dissimilarity_sampling', 'uncertainty', 'random', 'patch_based_selection']
python3 main_multi_view.py -MV_FLAG=TRAIN -MV_TYPE=MVT -QUERIES_STRATEGY=patch_based_selection -DATA_SET=M40v2
