# ESMTrack

The official implementation of **ESMTrack**.


## Install the environment
```
conda create -n esmtrack python=3.8
conda activate esmtrack
bash install.sh
```


## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Training
Download pre-trained [DropMAE ViT-Base weights](https://drive.google.com/file/d/1qMuBJtNIQQ-NCz98Pig72YVKQdasc49h/view?usp=share_link) and put it under `$PROJECT_ROOT$/pretrained_networks`.

```
python tracking/train.py \
--script esmtrack --config dropmae_256_150ep \
--save_dir ./output \
--mode multiple --nproc_per_node 2 \
--use_wandb 1
```

Replace `--config` with the desired model config under `experiments/esmtrack`.

We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Test and Evaluation

- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py esmtrack dropmae_256_150ep --dataset lasot --runid 150 --threads 8 --num_gpus 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py esmtrack dropmae_256_got_60ep --dataset got10k_test  --runid 60 --threads 8 --num_gpus 2
python lib/test/utils/transform_got10k.py --tracker_name esmtrack --cfg_name dropmae_256_got_60ep_060
```
- TrackingNet
```
python tracking/test.py esmtrack baseline --dataset trackingnet  --runid 150 --threads 8 --num_gpus 2
python lib/test/utils/transform_trackingnet.py --tracker_name esmtrack --cfg_name dropmae_256_150ep
```

For OTB and VOT2018 datasets, we use (eval_vot18_otb.sh) [PySOT-toolkit](https://github.com/StrangerZhang/pysot-toolkit) library for performance evaluation.

## Test FLOPs, and Speed

```
python tracking/profile_model.py --script esmtrack --config dropmae_256_150ep
```


## Acknowledgments
* This code is built upon [SSTrack](https://arxiv.org/abs/2507.21606), [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) and [PySOT-toolkit](https://github.com/StrangerZhang/pysot-toolkit). Thanks for their great work.
