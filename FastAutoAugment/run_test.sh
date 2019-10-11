root_dir=
project_dir=${root_dir}/mixnet-quant
dataset_dir=${root_dir}/dataset

now=$(date +"%Y%m%d_%H%M%S")
python3  train.py \
	--dataroot ${dataset_dir} \
        -c confs/mixnet_m.yaml \
	--pretrained=${project_dir}/mini_mixnet_m_checkpoint.pth.tar \
        --only_eval \
        --tag=test
