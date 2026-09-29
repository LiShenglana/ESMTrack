# Train on LasHeR
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
--script esmtrack --config dropmae_256_150ep \
--save_dir ./output \
--mode multiple --nproc_per_node 2 \
--use_wandb 0


# Test on RGB-T benchmarks (lasher, rgbt234, rgbt210, gtot, vtuav)
CUDA_VISIBLE_DEVICES=0,1 python tracking/test.py \
--tracker_name esmtrack --tracker_param dropmae_256_150ep \
--load_dir norm_cls_token_float --runid 25 \
--dataset_name lasher --threads 8 --num_gpus 2
