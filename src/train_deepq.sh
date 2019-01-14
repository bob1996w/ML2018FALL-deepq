#!/bin/sh
# argument 1 = train.csv
# argument 2 = image folder path/
# models are saved under this folder

PYTHONHASHSEED=0 python3 final_train.py model_169_1_e2_r $1 $2 169 2 8596
PYTHONHASHSEED=0 python3 final_train.py model_169_3_e1_r $1 $2 169 1 22127
PYTHONHASHSEED=0 python3 final_train.py model_169_4_e1_r $1 $2 169 1 28796
PYTHONHASHSEED=0 python3 final_train.py model_169_5_e2_r $1 $2 169 2 9558
PYTHONHASHSEED=0 python3 final_train.py model_169_7_e1_r $1 $2 169 1 19995
PYTHONHASHSEED=0 python3 final_train.py model_169_10_e2_r $1 $2 169 2 24003

PYTHONHASHSEED=0 python3 final_train.py model_121_3_e2_r $1 $2 121 2 820207
PYTHONHASHSEED=0 python3 final_train.py model_121_4_e2_r $1 $2 121 2 19930207
PYTHONHASHSEED=0 python3 final_train.py model_121_5_e1_r $1 $2 121 1 27194
PYTHONHASHSEED=0 python3 final_train.py model_121_8_e2_r $1 $2 121 2 10801
PYTHONHASHSEED=0 python3 final_train.py model_121_9_e2_r $1 $2 121 2 15350
PYTHONHASHSEED=0 python3 final_train.py model_121_10_e2_r $1 $2 121 2 8596


PYTHONHASHSEED=0 python3 final_train.py model_121_4_e2_STB $1 $2 121 2 24860
PYTHONHASHSEED=0 python3 final_train.py model_121_6_e1_STB $1 $2 121 1 17760

PYTHONHASHSEED=0 python3 final_train.py model_201_1_e2_s $1 $2 201 2 21562
PYTHONHASHSEED=0 python3 final_train.py model_201_2_e2_s $1 $2 201 2 12187
PYTHONHASHSEED=0 python3 final_train.py model_201_3_e2_s $1 $2 201 2 10549
