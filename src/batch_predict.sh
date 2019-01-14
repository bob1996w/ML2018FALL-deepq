# argument 1 = test.csv
# argument 2 = image folder
for filename in ./model_*; 
do 
    # extract base filename
    basefilename=${filename##*/}
    # delete extension for output filename
    basefilename=${basefilename%.*}
    echo ${basefilename}
    python3 test_for_batch.py $filename "result_${basefilename}.csv" $1 $2
done
