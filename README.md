# Official Implementation for the paper "Smoothed Embeddings for Certified Few-Shot Learning"

https://arxiv.org/abs/2202.01186

For experimental evaluation several ProtoNet models were trained on CUB200-2011, CIFAR-FS or mini-ImageNet. Pre-trained model weights are available [here](https://drive.google.com/drive/folders/1XWrB6-VYq14GcLJ9wrKBHUkcOz-Ifz5Q?usp=sharing) The training process can be reproduced by  obtaining a copy of the dataset and running:
```
cd code
python protonet/train.py --dataset DATASET --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr CL_PI_TR --iterations IT --num_support_tr N_SUP_TR --num_support_val N_SUP_VAL --num_query_tr N_QUE_TR --epochs N_EPOCHS --cuda
```
<strong>cOptions:sh</strong>
- DATASET: either 'cub200', 'cifar-fs' or 'mini-imagenet'
- PATH_TO_IMAGES: path to the folder, containing images
- PATH_TO_SPLITS: path to the folder, containing splits to classes in either .txt or .csv format
- PATH_TO_CHECKPOINT: path to the folder where the results of training would be saved
- CL_PI_TR: number of classes per batch for training, standard number is 5
- IT: number of iterations per epoch, standard number is 1000
- N_SUP_TR: number of support examples per class in batch for training (1 for 1-shot and 5 for 5-shot)
- N_SUP_VAL: number of support examples per class in batch for validation
- N_QUE_TR: number of query examples per class in batch for training, default is 15
- N_QUE_VAL: number of query examples per class in batch for validation, default is 15

For testing certified accuracy of smoothed models on test set, update <strong>certification.sh</strong> with pathes to datasets and trained models as well as <strong>certify_example.sh</strong> with the desired dataset setting and then run:
```
bash certify_example.sh
```
To visualize the results just run code/visualize.py
