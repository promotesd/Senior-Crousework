# DeMoE: Domain-Enhanced Mixture-of-Experts Adapter for Parameter-Efficient Remote Sensing Image-Text Retrieval


## Highlights

The goal of this work is to enhance implicit fine-grained knowledge transferring,  offering the best trade-off between performance and parameter efficiency.

## Usage
### Requirements
we use single NVIDIA 4090 24G GPU for training and evaluation. 
```
pytorch 1.12.1
torchvision 0.13.1
prettytable
easydict
```

### Prepare Datasets
Download the RSICD dataset from [here](https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file), RSITMD dataset from [here](https://www.modelscope.cn/datasets/YepingZhao/RSITMD),Sydney_captions form [here](https://www.modelscope.cn/datasets/YepingZhao/Sydney-Captions) and UCM_captions form [here](https://aistudio.baidu.com/datasetdetail/90740)


Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <RSICD>/
|       |-- imgs
|            |-- train 
|            |-- val
|            |-- test
|       |-- rsicd_train.json
|       |-- rsicd_val.json
|       |-- rsicd_test.json
|
|   |-- <RSITMD>/
|       |-- imgs
|            |-- train 
|            |-- val
|            |-- test
|       |-- rsitmd_train.json
|       |-- rsitmd_val.json
|       |-- rsitmd_test.json
|
|   |-- <Sydney_captions>/
|       |-- imgs
|       |-- data_captions.json
|
|   |-- <UCM_captions>/
|       |-- imgs
|       |-- data_captions.json
```


## Training

```python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+aux' \
--num_epoch 60 \
--root_dir '.../dataset_reid' \
--lr 3e-4 \
--num_experts 6 \
--topk 2 \
--reduction 8
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```


## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate for their contributions.