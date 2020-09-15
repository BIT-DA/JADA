#!/usr/bin/env bash

# A->W D->W W->D lr:0.0003 max_iter:20000
python train_office.py --gpu 0 --output_path checkpoint/ --log_file aw --data_set office --source_path ../data/office/amazon_list.txt --target_path ../data/office/webcam_list.txt
python train_office.py --gpu 0 --output_path checkpoint/ --log_file dw --data_set office --source_path ../data/office/dslr_list.txt --target_path ../data/office/webcam_list.txt
python train_office.py --gpu 0 --output_path checkpoint/ --log_file wd --data_set office --source_path ../data/office/webcam_list.txt --target_path ../data/office/dslr_list.txt
