# FlowDeltaQA

As we only modify the Flow part in [FlowQA](https://arxiv.org/abs/1810.06683), we keep the training / testing procedure as the same to [FlowQA released code](https://github.com/momohuang/FlowQA). 

Below we copy-and-paste the README from FlowQA released code.

#### Step 1:
perform the following:
```shell
pip install -r requirements.txt
```
to install all dependent python packages.

#### Step 2:
download necessary files using:
```shell
./download.sh
```

#### Step 3:
preprocess the data files using:
```shell
python preprocess_QuAC.py
python preprocess_CoQA.py
```

#### Step 4:
run the training code using:
```shell
python train_QuAC.py
python train_CoQA.py
```
For naming the output model, you can do
```shell
python train_OOOO.py --name XXX
```
Remove any answer marking by:
```shell
python train_OOOO.py --explicit_dialog_ctx 0
```
`OOOO` is the name of the dataset (QuAC or CoQA).

#### Step 5:
Do prediction with answer thresholding using
```shell
python predict_OOOO.py -m models_XXX/best_model.pt --show SS
```
`XXX` is the name you used during train.py.  
`SS` is the number of dialog examples to be shown.  
`OOOO` is the name of the dataset (QuAC or CoQA).
