# Certified Robustness for Few-Shot learning

## Experiments
This section contains code for reproducing the experiments.

For tranining the prototypical model on CUB200-2011, CIFAR-FS or mini-ImageNet in 1-shot and 5-shot setup, obtain a copy of the dataset and run:
```
cd code
python protonet/train.py --dataset cub200 --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 15 --epochs 40 --cuda
python protonet/train.py --dataset cifar-fs --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 15 --epochs 40 --cuda
python protonet/train.py --dataset mini-imagenet --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 15 --epochs 40 --cuda

python protonet/train.py --dataset cub200 --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 5 --num_support_val 5 --num_query_tr 15 --epochs 40 --cuda
python protonet/train.py --dataset cifar-fs --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 5 --num_support_val 5 --num_query_tr 15 --epochs 40 --cuda
python protonet/train.py --dataset mini-imagenet --dataset_root PATH_TO_IMAGES --splits_root PATH_TO_SPLITS --experiment_root PATH_TO_CHECKPOINT  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 5 --num_support_val 5 --num_query_tr 15 --epochs 40 --cuda
```

For testing certified accuracy of smoothed models on test set, update with pathes to datasets and trained models <strong>certification.sh</strong> as well as <strong>certify_example.sh</strong> with the desired dataset setting and then run:
```
bash certify_example.sh
```
To visualize the results just run code/visualize.py
