#!/bin/sh
# argument 1 = test.csv
# argument 2 = folder path/
# argument 3 = output.csv

### download model files to here
echo ">>> DOWNLOADING MODELS..."
# wget ...
wget https://www.dropbox.com/s/0rjxf9qqajxfp1t/model_121_3_e2_r.h5?dl=1
wget https://www.dropbox.com/s/gd60syzqaubyknp/model_121_4_e2_r.h5?dl=1
wget https://www.dropbox.com/s/76tkzaiu5n5b1fx/model_121_4_e2_STB.h5?dl=1
wget https://www.dropbox.com/s/vks4rjciczn6fb5/model_121_5_e1_r.h5?dl=1
wget https://www.dropbox.com/s/hq7q8zomiywp93q/model_121_6_e1_STB.h5?dl=1
wget https://www.dropbox.com/s/tj1zkgmitep58eb/model_121_8_e2_r.h5?dl=1
wget https://www.dropbox.com/s/i75xgmj2ce9u472/model_121_9_e2_r.h5?dl=1
wget https://www.dropbox.com/s/7lqljlelg968p39/model_121_10_e2_r.h5?dl=1
wget https://www.dropbox.com/s/5aj4lik360qp6pz/model_169_1_e2_r.h5?dl=1
wget https://www.dropbox.com/s/hzxhvisie14wxqh/model_169_3_e1_r.h5?dl=1
wget https://www.dropbox.com/s/judthpoxdh7s9yt/model_169_4_e1_r.h5?dl=1
wget https://www.dropbox.com/s/xxdoxoe8iyl52nv/model_169_5_e2_r.h5?dl=1
wget https://www.dropbox.com/s/16wls4rbzv9j5d3/model_169_7_e1_r.h5?dl=1
wget https://www.dropbox.com/s/cc737apgk5zu4io/model_169_10_e2_r.h5?dl=1
wget https://www.dropbox.com/s/8d3lfe7k1myaej5/model_201_1_e2_s.h5?dl=1
wget https://www.dropbox.com/s/wqg3iimh0ezgnla/model_201_2_e2_s.h5?dl=1
wget https://www.dropbox.com/s/a30lrblkojv8i05/model_201_3_e2_s.h5?dl=1

### predict models to this folder
echo ">>> PREDICTING MODELS..."
bash ./batch_predict.sh $1 $2

### pca 
echo ">>> PERFORMING PCA's.."
python3 reproduce_pca_fix.py result_model_169_3_e1_r.csv result_model_121_3_e2_r.csv class0.csv
python3 reproduce_pca_fix.py result_model_169_5_e2_r.csv result_model_201_2_e2_s.csv class1.csv
python3 reproduce_pca_fix.py result_model_169_7_e1_r.csv result_model_121_4_e2_STB.csv class2.csv
python3 reproduce_pca_fix.py result_model_201_3_e2_s.csv result_model_121_3_e2_r.csv class3.csv
python3 reproduce_pca_fix.py result_model_169_10_e2_r.csv result_model_201_2_e2_s.csv result_model_169_5_e2_r.csv class4.csv
python3 reproduce_pca_fix.py result_model_169_5_e2_r.csv result_model_201_1_e2_s.csv class5.csv
python3 reproduce_pca_fix.py result_model_169_1_e2_r.csv result_model_121_4_e2_r.csv result_model_201_2_e2_s.csv class6.csv
python3 reproduce_pca_fix.py result_model_121_4_e2_STB.csv result_model_201_2_e2_s.csv result_model_121_4_e2_r.csv class7.csv
python3 reproduce_pca_fix.py result_model_201_2_e2_s.csv result_model_121_6_e1_STB.csv class8.csv
python3 reproduce_pca_fix.py result_model_169_7_e1_r.csv result_model_169_4_e1_r.csv result_model_201_2_e2_s.csv class9.csv
python3 reproduce_pca_fix.py result_model_169_10_e2_r.csv result_model_121_4_e2_STB.csv class10.csv
python3 reproduce_pca_fix.py result_model_121_8_e2_r.csv result_model_169_5_e2_r.csv result_model_201_2_e2_s.csv result_model_169_4_e1_r.csv class11.csv
python3 reproduce_pca_fix.py result_model_121_5_e1_r.csv result_model_121_4_e2_STB.csv result_model_121_4_e2_r.csv class12.csv
python3 reproduce_pca_fix.py result_model_121_10_e2_r.csv result_model_121_4_e2_STB.csv result_model_121_9_e2_r.csv class13.csv

python3 reproduce_pca_fix.py class0.csv class1.csv class2.csv class0_2.csv
python3 reproduce_pca_fix.py class3.csv class4.csv class5.csv class3_5.csv
python3 reproduce_pca_fix.py class6.csv class7.csv class8.csv class6_8.csv
python3 reproduce_pca_fix.py class9.csv class10.csv class11.csv class12.csv class9_12.csv

python3 reproduce_pca_fix.py class0_2.csv class3_5.csv class6_8.csv class9_12.csv class0_12.csv


# merge
echo ">>> MERGING FILES..."
# output: merged_predicts.csv
python3 FinalProject_CombineResult.py merged_predicts.csv

# minmax
echo ">>> ADJUST VALUES..."
python3 MinMax.py merged_predicts.csv > $3

# output
echo ">>> DONE. Output file is $3"
