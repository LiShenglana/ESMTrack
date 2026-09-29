# ESMTrack

The official implementation of **ESMTrack**.

[[Models (Hugging Face)](https://huggingface.co/ShenglanLiaaa/ESMTrack)]


## Model Weights
Checkpoints are hosted on [Hugging Face](https://huggingface.co/ShenglanLiaaa/ESMTrack). Download the checkpoints into the project root:
```
hf download ShenglanLiaaa/ESMTrack --include "checkpoints/*" --local-dir .
```

| Checkpoint | Benchmark |
|:--|:--|
| `checkpoints/LasHeR_best_checkpoint.pth` | LasHeR |
| `checkpoints/VTUAV_best_checkpoint.pth` | VTUAV |
| `checkpoints/GTOT_best_checkpoint.pth` | GTOT |
| `checkpoints/RGBT210_best_checkpoint.pth` | RGBT210 |
| `checkpoints/RGBT234_best_checkpoint.pth` | RGBT234 |


## Install the environment
```
conda create -n esmtrack python=3.8
conda activate esmtrack
bash install.sh
```


## Data Preparation
ESMTrack is trained on [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR) and evaluated on the RGB-T benchmarks LasHeR, RGBT234, RGBT210, GTOT and VTUAV. Put the datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasher
            |-- trainingset
                |-- 2up
                    |-- visible        # v000000.jpg, v000001.jpg, ...
                    |-- infrared       # i000000.jpg, i000001.jpg, ...
                    |-- init.txt
                ...
            |-- testingset
                |-- list.txt
                |-- 10runone
                    |-- visible
                    |-- infrared
                    |-- init.txt
                ...
        -- rgbt234
            |-- list.txt
            |-- afterrain
                |-- visible
                |-- infrared
                |-- visible.txt
            ...
        -- rgbt210
            |-- list.txt
            |-- afterrain
                |-- visible
                |-- infrared
                |-- init.txt
            ...
        -- gtot
            |-- list.txt
            |-- BlackCar
                |-- v
                |-- i
                |-- groundTruth_v.txt
            ...
        -- vtuav
            |-- VTUAV-ST.txt
            |-- animal_001
                |-- rgb
                |-- ir
                |-- rgb.txt
            ...
   ```
The LasHeR training sequences used are listed in `lib/train/data_specs/lasher_all.txt`.


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
Download the pre-trained [DropMAE ViT-Base weights](https://drive.google.com/file/d/1qMuBJtNIQQ-NCz98Pig72YVKQdasc49h/view?usp=share_link) (`dropmae_k700_800E.pth`, K700-800E) released by the [DropMAE authors](https://github.com/jimmy-dq/DropMAE) and put it under `$PROJECT_ROOT$/pretrained_networks`. These weights are not redistributed in this repository or on Hugging Face.

Train on LasHeR (`DATA.TRAIN.DATASETS_NAME: LasHeR_all` in the config):
```
python tracking/train.py --script esmtrack --config dropmae_256_150ep --save_dir ./output --mode multiple --nproc_per_node 2 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/esmtrack`.

We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Test and Evaluation
Run the tracker on an RGB-T benchmark with the released checkpoint of that benchmark. `--dataset_name` can be `lasher`, `rgbt234`, `rgbt210`, `gtot` or `vtuav`:
```
python tracking/test.py --tracker_name esmtrack --tracker_param dropmae_256_150ep --checkpoint checkpoints/LasHeR_best_checkpoint.pth --dataset_name lasher --threads 8 --num_gpus 2
```
Results are saved to `output/test/tracking_results/esmtrack/dropmae_256_150ep/LasHeR_best_checkpoint/<dataset>/`.

To test a checkpoint trained by yourself, use `--load_dir` and `--runid` instead of `--checkpoint`; the checkpoint is then loaded from `output/checkpoints/train/esmtrack/<tracker_param>/<load_dir>/ESMTrack_ep<runid>.pth.tar`.

The saved result files can be evaluated with the official toolkits of each benchmark (e.g. the [LasHeR toolkit](https://github.com/BUGPLEASEOUT/LasHeR)).

## Test FLOPs, and Speed

```
python tracking/profile_model.py --script esmtrack --config dropmae_256_150ep
```

## Acknowledgments
* This code is built upon [SSTrack](https://arxiv.org/abs/2507.21606), [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) and [PySOT-toolkit](https://github.com/StrangerZhang/pysot-toolkit). Thanks for their great work.
