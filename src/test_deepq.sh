#!/bin/sh
# argument 1 = test.csv
# argument 2 = folder path/
# argument 3 = output.csv

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
