root_dir=
project_dir=${root_dir}/
dataset_dir=${root_dir}/

python3  train.py \
	--dataroot ${dataset_dir} \
    -c confs/mixnet_m_quant.yaml \
    --dataset cifar100 \
	--pretrained=${project_dir}/mixnet_m_checkpoint.pth.tar \
	--save=${project_dir}/ \
    --tag=
