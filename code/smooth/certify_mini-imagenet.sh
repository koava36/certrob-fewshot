# 1shot
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/1shot/N3000_sigma1.0_a0.01.txt 1 --max 20 --N 3000 --K 5 --alpha 0.01 --dataset mini-imagenet --cuda_number 2
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/1shot/N3000_sigma1.0_a0.001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
#  python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/1shot/N3000_sigma1.0_a0.0001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.0001 --dataset mini-imagenet --cuda_number 2

# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 0.25 ../data/certify/mini-imagenet/1shot/N3000_sigma0.25_a0.001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet--cuda_number 2
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 0.5 ../data/certify/mini-imagenet/1shot/N3000_sigma0.5_a0.001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2

python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/1shot/N500_sigma1.0_a0.001.txt 1 --max 1 --N 500 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/1shot/N1000_sigma1.0_a0.001.txt 1 --max 1 --N 1000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2


# # 5shot

# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_5shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/5shot/N3000_sigma1.0_a0.01.txt 5 --max 20 --N 3000 --K 5 --alpha 0.01 --dataset mini-imagenet --cuda_number 2
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_5shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/5shot/N3000_sigma1.0_a0.001.txt 5 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
#  python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_5shot_normalize_512_augment_1 1.0 ../data/certify/mini-imagenet/5shot/N3000_sigma1.0_a0.0001.txt 5 --max 20 --N 3000 --K 5 --alpha 0.0001 --dataset mini-imagenet --cuda_number 2

# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_5shot_normalize_512_augment_1 0.25 ../data/certify/mini-imagenet/5shot/N3000_sigma0.25_a0.001.txt 5 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_5shot_normalize_512_augment_1 0.5 ../data/certify/mini-imagenet/5shot/N3000_sigma0.5_a0.001.txt 5 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2

# # 1shot smoothed on train

# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_smooth_1 0.25 ../data/certify/mini-imagenet/1shot/smoothed_tr/N3000_sigma0.25_a0.001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_smooth_1 0.5 ../data/certify/mini-imagenet/1shot/smoothed_tr/N3000_sigma0.5_a0.001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
# python certify.py ../../raid/data/datasets/miniimagenet/miniimagenet/data ../../raid/data/datasets/miniimagenet/miniimagenet/splits/ravi-larochelle ../../raid/data/okuznetsova/mini-imagenet/protonet_1shot_normalize_512_smooth_1 1.0 ../data/certify/mini-imagenet/1shot/smoothed_tr/N3000_sigma1.0_a0.001.txt 1 --max 20 --N 3000 --K 5 --alpha 0.001 --dataset mini-imagenet --cuda_number 2
