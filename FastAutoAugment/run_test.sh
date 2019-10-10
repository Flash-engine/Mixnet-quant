root_dir=
project_dir=${root_dir}/
dataset_dir=${root_dir}/

python3  train.py \
	--dataroot ${dataset_dir} \
    -c confs/mixnet_m.yaml \
    --dataset cifar100 \
	--save=${project_dir}/ \
    --only_eval \
    --tag=test 
