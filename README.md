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

For testing model and getting embedding risks on test images, run:
```
cd code
python certify.py cub200 PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cub200/1shot/N1000/sigma0.25_a0.001.txt 1 0.25 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py cub200 PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cub200/1shot/N1000/sigma0.5_a0.001.txt 1 0.5 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py cub200 PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cub200/1shot/N1000/sigma1.0_a0.001.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py cub200 PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cub200/1shot/N1000/sigma1.0_a0.01.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.01
python certify.py cub200 PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cub200/1shot/N1000/sigma1.0_a0.0001.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.0001

python certify.py cifar-sf PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cifar-fs/1shot/N1000/sigma0.25_a0.001.txt 1 0.25 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py cifar-sf PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cifar-fs/1shot/N1000/sigma0.5_a0.001.txt 1 0.5 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py cifar-sf PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cifar-fs/1shot/N1000/sigma1.0_a0.001.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py cifar-sf PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cifar-fs/1shot/N1000/sigma1.0_a0.01.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.01
python certify.py cifar-sf PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/cifar-fs/1shot/N1000/sigma1.0_a0.0001.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.0001

python certify.py mini-imagenet PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/mini-imagenet/1shot/N1000/sigma0.25_a0.001.txt 1 0.25 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py mini-imagenet PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/mini-imagenet/1shot/N1000/sigma0.5_a0.001.txt 1 0.5 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py mini-imagenet PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/mini-imagenet/1shot/N1000/sigma1.0_a0.001.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.001
python certify.py mini-imagenet PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/mini-imagenet/1shot/N1000/sigma1.0_a0.01.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.01
python certify.py mini-imagenet PATH_TO_IMAGES PATH_TO_SPLITS PATH_TO_CHECKPOINT ../data/certify/mini-imagenet/1shot/N1000/sigma1.0_a0.0001.txt 1 1.0 --max 20 --N 1000 --K 10 --alpha 0.0001
```
