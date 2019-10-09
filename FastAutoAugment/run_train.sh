partition=VI_IPS_1080TI
gpu_num=8
task_num=1

root_dir=/mnt/lustre/zhangyao/micronet
project_dir=${root_dir}/mini-mixnet-m
dataset_dir=${root_dir}/dataset

now=$(date +"%Y%m%d_%H%M%S")
srun    --partition=${partition} \
	--mpi=pmi2 \
	--gres=gpu:$gpu_num \
	-n$task_num\
	--ntasks-per-node=$gpu_num \
	--job-name=MicroNet \
	--kill-on-bad-exit=0 \
	python3  train.py \
	--dataroot ${dataset_dir} \
        -c confs/mixnet_m.yaml \
        --dataset cifar100 \
	--save ${project_dir}/mixnet_m_checkpoint.pth.tar \
        --tag mini_0.0.4
