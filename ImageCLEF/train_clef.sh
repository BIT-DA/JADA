#!/usr/bin/env bash

# i->p p->i i->c c->i c->p p->c lr:0.001 max_iter:10000
python train_clef.py --gpu 1 --output_path checkpoint/ --log_file ip --data_set clef --source_path ../data/clef/i_list.txt --target_path ../data/clef/p_list.txt
python train_clef.py --gpu 1 --output_path checkpoint/ --log_file pi --data_set clef --source_path ../data/clef/p_list.txt --target_path ../data/clef/i_list.txt
python train_clef.py --gpu 1 --output_path checkpoint/ --log_file ic --data_set clef --source_path ../data/clef/i_list.txt --target_path ../data/clef/c_list.txt
python train_clef.py --gpu 1 --output_path checkpoint/ --log_file ci --data_set clef --source_path ../data/clef/c_list.txt --target_path ../data/clef/i_list.txt
python train_clef.py --gpu 1 --output_path checkpoint/ --log_file cp --data_set clef --source_path ../data/clef/c_list.txt --target_path ../data/clef/p_list.txt
python train_clef.py --gpu 1 --output_path checkpoint/ --log_file pc --data_set clef --source_path ../data/clef/p_list.txt --target_path ../data/clef/c_list.txt
