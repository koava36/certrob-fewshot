# # train base protonet for 1 shot on cub200
#  python train.py --experiment_root ../../raid/data/okuznetsova/protonet_1shot_normalize_512_augment_1 --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 15 --epochs 10 --cuda --cuda_number 6 --manual_seed 7

# # train base protonet for 5 shot on cub200
#  python train.py --experiment_root ../../raid/data/okuznetsova/protonet_5shot_normalize_512_augment_1 --classes_per_it_tr 5 --iterations 1000 --num_support_tr 5 --num_support_val 5 --num_query_tr 15 --epochs 10 --cuda --cuda_number 6 --manual_seed 7

# # train base protonet for 1 shot with smoothing on cub200
# python train.py --experiment_root ../../raid/data/okuznetsova/protonet_1shot_normalize_512_smooth_1 --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 5 --smooth_samples 1 --sigma_train 1.0 --epochs 10 --cuda --cuda_number 6 --manual_seed 7

# #train base protonet for 1 shot on mini-imagenet
#  python train.py --experiment_root ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 --dataset mini-imagenet --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 15 --epochs 25 --cuda --cuda_number 2 --manual_seed 7

# # train base protonet for 5 shot on mini-imagenet
#  python train.py --experiment_root ../../raid/data/okuznetsova/mini-imagenet/protonet_5shot_normalize_512_augment_1 --dataset mini-imagenet --classes_per_it_tr 5 --iterations 1000 --num_support_tr 5 --num_support_val 5 --num_query_tr 15 --epochs 25 --cuda --cuda_number 2 --manual_seed 7

# train base protonet for 1 shot with smoothing on mini-imagenet
# python train.py --experiment_root ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_smooth_1 --dataset mini-imagenet --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 5 --smooth_samples 5 --sigma_train 1.0 --epochs 25 --cuda --cuda_number 2 --manual_seed 7

# train base protonet for 1 shot on cifar-fs
python protonet/train.py --dataset cifar-fs --dataset_root ../../raid/data/datasets/cifar100/cifar-100-python/cifar-fs --splits_root ../../raid/data/datasets/cifar100/cifar-100-python/cifar-fs-splits --experiment_root ../../raid/data/okuznetsova/cifar-fs/protonet_1shot_normalize_512_augment_test  --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 15 --epochs 1 --cuda --cuda_number 4 --manual_seed 7

# # train base protonet for 5 shot on mini-imagenet
#  python train.py --experiment_root ../../raid/data/okuznetsova/cifar-fs/protonet_5shot_normalize_512_augment_1 --dataset cifar-fs --classes_per_it_tr 5 --iterations 1000 --num_support_tr 5 --num_support_val 5 --num_query_tr 15 --epochs 40 --cuda --cuda_number 4 --manual_seed 7

# # train base protonet for 1 shot with smoothing on mini-imagenet
# python train.py --experiment_root ../../raid/data/okuznetsova/cifar-fs/protonet_1shot_normalize_512_smooth_1 --dataset cifar-fs --classes_per_it_tr 5 --iterations 1000 --num_support_tr 1 --num_support_val 1 --num_query_tr 5 --smooth_samples 10 --sigma_train 1.0 --epochs 40 --cuda --cuda_number 4 --manual_seed 7