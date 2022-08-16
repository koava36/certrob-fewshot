source certification.sh

outdir=../data/certify/cub/1shot
nshot=1
sigma=1.0
alpha=0.0001
N=1000
device=3

python certify.py cub200 $cub200_datapath $cub200_splitpath $cub200_1shot_model $outdir $nshot $sigma --max 20 --N $N --K 100 --alpha $alpha --cuda_number $device

