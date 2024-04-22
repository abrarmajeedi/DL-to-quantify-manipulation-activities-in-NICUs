

# ActionFormer for Localizing Manipulation Activities in NICU videos.

Full instructions and details for ActionFormer can be found at the [official repo](https://github.com/happyharrycn/actionformer_release/blob/main/README.md).

## Setup:
Follow directions in INSTALL.md to setup ActionFormer on your machine. 


## Data Annotation format:
Your training/val data annotations need to be in the same format as the *sample_data/annotations.json* file. 

## Config:
Modify the paths in the sample config file (or it copy), adding the paths to your feature directory (*feat_folder*, *file_ext* etc),  data annotation file (*json_file*), feature extraction parameters (such as *stride*, *num_frames* i.e. with feature window size etc). Also provide the number of classes you are trying to detect and their label names in the *label_dict* field. You can also specify the path where your trained model weights will be saved (*output_folder*). Other parameters can also be changed as needed.

## Training:
The model is trained using the following command:

```shell
python -u ./train.py ./configs/your_config.yaml
```

The trained model will be saved in ./ckpt or your specified path.

## Eval
The trained model can be evaluated using the following command:
```shell
python  ./eval.py  ./configs/your_config.yaml ./path_to_ckpt_folder/
```
This command will evaluate the last epoch of your training, howvere you can proovide the path to the exact epoch to specifically evaluate any epoch:
```shell
python  ./eval.py  ./configs/your_config.yaml ./path_to_ckpt/epoch_010.tar
```
This code will evaluate your model generating the precision, recall and mAP scores on your test/val set.


## Making predictions

In order to make predictions, simply run the following command:
```shell
python  ./eval.py  ./configs/your_config.yaml ./path_to_ckpt_folder/ --saveonly
```

The trained model will be run on your test set, and the predictions (i.e. detected actions and their time stamps) will be dumped in the *./path_to_ckpt_folder/* as a pickle file. 




