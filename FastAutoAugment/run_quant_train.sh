partition=VI_SP_Z_M40
gpu_num=4
task_num=1

root_dir=/mnt/lustre/liuxin/workspace/MicroNetChallenge
project_dir=${root_dir}/mini-mixnet-m
dataset_dir=${root_dir}/dataset/cifar100

now=$(date +"%Y%m%d_%H%M%S")
srun    --partition=${partition} \
	--mpi=pmi2 \
	--gres=gpu:$gpu_num \
	-n$task_num\
	--ntasks-per-node=$gpu_num \
	--job-name=DSQ_quant \
	--kill-on-bad-exit=0 \
	python3  train.py \
	--dataroot ${dataset_dir} \
    -c confs/mixnet_m.yaml \
    --dataset cifar100 \
	--pretrained=${project_dir}/mixnet_m_checkpoint.pth.tar \
	--save=${project_dir}/DSQ_quant/mixnet_dsq_0.001_192_weight_int4_only_checkpoint.pth.tar \
    --tag mini_dsq_0.001_192_weight_int4_only
