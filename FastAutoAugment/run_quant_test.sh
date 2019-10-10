root_dir=
project_dir=${root_dir}/
dataset_dir=${root_dir}/dataset/

python3  train.py \
	--dataroot ${dataset_dir} \
    -c confs/mixnet_m_quant.yaml \
    --dataset cifar100 \
	--save=${project_dir}/DSQ_quant/mixnet_dsq_0.0001_128_checkpoint.pth.tar \
    --only_eval \
    --tag=test_quant 
