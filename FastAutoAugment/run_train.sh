root_dir=
project_dir=${root_dir}/mixnet-quant
dataset_dir=${root_dir}/dataset

now=$(date +"%Y%m%d_%H%M%S")
python3  train.py \
	--dataroot ${dataset_dir} \
        -c confs/mixnet_m.yaml \
	--save=${project_dir}/mini_mixnet_m_checkpoint.pth.tar \
        --tag mini_mixnet_0.0.1
