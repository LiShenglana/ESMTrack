#!/bin/bash

#cd ..
#cd ..
#cd tracking/
#python test.py --dataset_name lasher --runid 20 --load_dir grounding_search_triplet_cross_seq
#cd ..
#
#cd lib/train
check_name="norm_cls_token_float"
# 第一组实验：基础参数
echo "Starting training..."
python run_training.py --save_dir /home/cscv/Documents/lsl/ESMTrackRGBT/checkpoints/train/esmtrack/dropmae_256_150ep/$check_name

# 第二组实验：调整学习率和损失权重 (假设你代码支持命令行参数)
echo "Starting testing..."

cd ..
cd ..
cd tracking/
python test.py --dataset_name gtot --runid 20 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 20 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 20 --load_dir $check_name
python test.py --dataset_name lasher --runid 20 --load_dir $check_name
python test.py --dataset_name gtot --runid 19 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 19 --load_dir $check_name
python test.py --dataset_name gtot --runid 21 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 21 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 22 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 23 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 24 --load_dir $check_name
python test.py --dataset_name rgbt234 --runid 25 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 19 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 21 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 22 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 23 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 24 --load_dir $check_name
python test.py --dataset_name rgbt210 --runid 25 --load_dir $check_name
python test.py --dataset_name gtot --runid 22 --load_dir $check_name
python test.py --dataset_name gtot --runid 23 --load_dir $check_name
python test.py --dataset_name gtot --runid 24 --load_dir $check_name
python test.py --dataset_name gtot --runid 25 --load_dir $check_name
python test.py --dataset_name lasher --runid 19 --load_dir $check_name
python test.py --dataset_name lasher --runid 25 --load_dir $check_name
python test.py --dataset_name lasher --runid 21 --load_dir $check_name
python test.py --dataset_name lasher --runid 22 --load_dir $check_name
python test.py --dataset_name lasher --runid 23 --load_dir $check_name
python test.py --dataset_name lasher --runid 24 --load_dir $check_name
python test.py --dataset_name vtuav --runid 25 --load_dir $check_name
python test.py --dataset_name vtuav --runid 21 --load_dir $check_name
python test.py --dataset_name vtuav --runid 22 --load_dir $check_name
python test.py --dataset_name vtuav --runid 23 --load_dir $check_name
python test.py --dataset_name vtuav --runid 24 --load_dir $check_name

#cd ..
#cd lib/train
#check_name="backward_iou_0.1"
## 第一组实验：基础参数
#echo "Starting training..."
#python run_training.py --save_dir /home/cscv/Documents/lsl/ESMTrackRGBT/checkpoints/train/esmtrack/dropmae_256_150ep/$check_name --backward_iou 0.1
#
## 第二组实验：调整学习率和损失权重 (假设你代码支持命令行参数)
#echo "Starting testing..."
#
#cd ..
#cd ..
#cd tracking/
#python test.py --dataset_name gtot --runid 20 --load_dir $check_name
#python test.py --dataset_name rgbt234 --runid 20 --load_dir $check_name
#python test.py --dataset_name rgbt210 --runid 20 --load_dir $check_name
#python test.py --dataset_name lasher --runid 20 --load_dir $check_name
#cd ..
#cd lib/train
#check_name="backward_iou_0.4"
## 第一组实验：基础参数
#echo "Starting training..."
#python run_training.py --save_dir /home/cscv/Documents/lsl/ESMTrackRGBT/checkpoints/train/esmtrack/dropmae_256_150ep/$check_name --backward_iou 0.4
#
## 第二组实验：调整学习率和损失权重 (假设你代码支持命令行参数)
#echo "Starting testing..."
#
#cd ..
#cd ..
#cd tracking/
#python test.py --dataset_name gtot --runid 20 --load_dir $check_name
#python test.py --dataset_name rgbt234 --runid 20 --load_dir $check_name
#python test.py --dataset_name rgbt210 --runid 20 --load_dir $check_name
#python test.py --dataset_name lasher --runid 20 --load_dir $check_name
#cd ..
#cd lib/train
#check_name="backward_iou_0.5"
## 第一组实验：基础参数
#echo "Starting training..."
#python run_training.py --save_dir /home/cscv/Documents/lsl/ESMTrackRGBT/checkpoints/train/esmtrack/dropmae_256_150ep/$check_name --backward_iou 0.5
#
## 第二组实验：调整学习率和损失权重 (假设你代码支持命令行参数)
#echo "Starting testing..."
#
#cd ..
#cd ..
#cd tracking/
#python test.py --dataset_name gtot --runid 20 --load_dir $check_name
#python test.py --dataset_name rgbt234 --runid 20 --load_dir $check_name
#python test.py --dataset_name rgbt210 --runid 20 --load_dir $check_name
#python test.py --dataset_name lasher --runid 20 --load_dir $check_name
###python test.py --dataset_name gtot --runid 19 --load_dir $check_name
##python test.py --dataset_name gtot --runid 20 --load_dir $check_name
##python test.py --dataset_name gtot --runid 21 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 19 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 20 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 21 --load_dir $check_name
##python test.py --dataset_name lasher --runid 20 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 19 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 20 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 21 --load_dir $check_name
##python test.py --dataset_name gtot --runid 22 --load_dir $check_name
##python test.py --dataset_name gtot --runid 23 --load_dir $check_name
##python test.py --dataset_name gtot --runid 24 --load_dir $check_name
##python test.py --dataset_name gtot --runid 25 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 22 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 23 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 24 --load_dir $check_name
##python test.py --dataset_name rgbt234 --runid 25 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 22 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 23 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 24 --load_dir $check_name
##python test.py --dataset_name rgbt210 --runid 25 --load_dir $check_name
#
echo "Training finished with status: $status" | mail -s "Training Done" shenglanli@cumt.edu.cn