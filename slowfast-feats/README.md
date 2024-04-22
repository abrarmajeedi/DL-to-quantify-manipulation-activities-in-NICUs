# Slowfast Feats

  

## Prerequisites

- Install the latest [PySlowFast library](https://github.com/facebookresearch/SlowFast) from FAIR and make sure it is working

- Clone this repository to your machine

- Note the official SlowFast library sometimes throws some errors based on your environment setup but some quick search on the issues sections and SlowFast forums to resolve.

  

## Dataset prep

- The frames are extracted on the fly, so we can use videos directly

- All the videos need to be in one directory

- You also need the vid_list file, which simply contains two columns, the video name(without the extension) and number of frames in that video. Note there is no header. (You can use the create_vid_list.py file to create that file.)

  

## Pretrained models:

- You can download the pretrained model and the corresponding config file from the [Slowfast model zoo](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) and copy it to your desired location.
 
  

## Extracting features

- Edit the tmp folder(make sure this directory exists), datapath, output directory and the pretrained model path in the config corresponding to the model you want to use.

- Our experiments used *./slowfast-feats/configs/SLOWFAST_8x8_R50.yaml*, so you can use that to set up your own copy of the config.

- Modify the parameters as needed e.g. *NUM_GPUS*, *PATH_TO_DATA_DIR*, *VID_LIST*, *TMP_FOLDER*, *FPS*, *SAMPLE_SIZE* (i.e. resolution), *OUTPUT_DIR*, *CHECKPOINT_FILE_PATH* etc.

- Then run 
```shell
python extract_feat.py --cfg configs/your_config_file
```
- Your features will be saved in the *OUTPUT_DIR* path you specified in the config file.
- This path will be required by the ActionFormer model.