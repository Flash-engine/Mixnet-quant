root_dir=
project_dir=${root_dir}/
dataset_dir=${root_dir}/dataset/cifar100

python3  train.py \
	--dataroot ${dataset_dir} \
    -c confs/mixnet_m.yaml \
    --dataset cifar100 \
	--pretrained=${project_dir}/mixnet_m_checkpoint.pth.tar \
	--save= \
    --tag= 
