# ML Final project-deepq readme

## Package needed

```
Keras==2.1.6
numpy==1.14.0
pandas==0.23.4
Pillow==5.3.0
scikit-learn==0.20.2
scipy==1.1.0
tensorflow==1.10.1
```

## For TA in ML class

### How to run test with trained models download

```bash
cd src
bash test.sh <test.csv> <image folder path/> <output.csv>
```

- image folder path has to end with `/`. 

## For HTC deepq

### 1. train

```bash
cd src
bash train_deepq.sh <train.csv> <image folder path/>
```

- image folder path has to end with `/`. 

### 2. test 

```bash
cd src
bash test_deepq.sh <test.csv> <image folder path/> <output.csv>
```

- image folder path has to end with `/`. 