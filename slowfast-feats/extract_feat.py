#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Modified to process a list of videos

"""Extract features for videos using pre-trained networks"""

import numpy as np
import torch
import os
import subprocess
import shutil
import time
import cv2
from tqdm import tqdm
import sys
import argparse
# print(sys.path)
# sys.path.append('/videos/code/slowfast-feats/SlowFast/')
# sys.path.append('/videos/code/slowfast-feats/SlowFast/slowfast')
# sys.path.append('/videos/code/slowfast-feats/SlowFast/slowfast/')
# sys.path.append('/videos/code/slowfast-feats/SlowFast/slowfast/datasets')
# print(sys.path)
import SlowFast.slowfast.utils.checkpoint as cu
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from feats import build_model, VideoDataset
from feats.custom_config import load_config

"""Wrapper to train and test a video classification model."""
#from slowfast.utils.parser import parse_args

torch.backends.cudnn.benchmark = True
logger = logging.get_logger(__name__)
cv2.setNumThreads(0)



@torch.no_grad()
def perform_inference(test_loader, model, cfg):
    """
    Perform mutli-view testing that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    feat_arr = tuple()
    for inputs in tqdm(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to('cuda:0',non_blocking=True)
                #print("input",i,inputs[i].shape)
                #inputs[i] = inputs[i].cuda(non_blocking=True).to(f'cuda:{model.device_ids[0]}')
        else:
            inputs = inputs.to('cuda:0',non_blocking=True)
            #inputs = inputs.cuda(non_blocking=True).to(f'cuda:{model.device_ids[0]}')
        model = model.to('cuda:0')
        # print([inp.shape for inp in inputs])
        feat = model(inputs)
        # print(type(feat))
        # import pdb;pdb.set_trace()
        feat = feat.cpu().numpy()
        feat_arr += (feat, )

    # concat all feats
    feat_arr = np.concatenate(feat_arr, axis=0)
    #print("feat_arr.shape",feat_arr.shape)
    return feat_arr


def decode_frames(cfg, vid_id):
    # get video file path
    video_file = (
        os.path.join(cfg.DATA.PATH_TO_DATA_DIR, vid_id) + cfg.DATA.VID_FILE_EXT
    )
    # create output folder
    output_folder = os.path.join(cfg.DATA.TMP_FOLDER, vid_id)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # ffmpeg cmd
    command = ['ffmpeg',
               '-i', '{:s}'.format(video_file),
               #'-vf', '\"hflip\"',
               '-r', '{:s}'.format(str(cfg.DATA.FPS)),
               #'-s', '{}x{}'.format(cfg.DATA.SAMPLE_SIZE[0],cfg.DATA.SAMPLE_SIZE[1]),
               '-f', 'image2', '-q:v', '1',
               '{:s}/%010d.jpg'.format(output_folder)
              ]
    command = ' '.join(command)
    #return True, None
    # call ffmpeg
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return False, err.output

    return True, None

def remove_frames(cfg, vid_id):
    output_folder = os.path.join(cfg.DATA.TMP_FOLDER, vid_id)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

def test(cfg):
    """
    Perform multi-view testing/feature extraction on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    model.cuda()
    # Enable eval mode.
    model.eval()
    if cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model,convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2", data_parallel=False)
    # switch to data parallel
    # print(model)
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(cfg.NUM_GPUS)])

    # print(model)

    vid_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)
    videos_list_file = os.path.join(cfg.DATA.VID_LIST)

    print("Loading Video List ...")
    with open(videos_list_file) as f:
        lines = [line.rstrip() for line in f]
    videos = []
    for line in lines:
        vid, num_frames = line.split(' ')
        videos.append((vid, int(num_frames)))
    print("Done")
    print("----------------------------------------------------------")

    if cfg.DATA.READ_VID_FILE:
        rejected_vids = []

    print("{} videos to be processed...".format(len(videos)))
    print("----------------------------------------------------------")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    start_time = time.time()
    # videos = videos[::-1]
    for vid_no, cur_video in enumerate(videos):
        vid, num_frames = cur_video
        # Create video testing loaders.
        path_to_vid = os.path.join(vid_root, os.path.split(vid)[0])
        vid_id = os.path.split(vid)[1]
        print("vid_id", vid_id)
        
        path = os.path.split(vid)[-1]
        out_path = os.path.join(cfg.OUTPUT_DIR, path)
        # out_file = vid_id.split(".")[0] + ".npz"
        out_file = out_path + ".npz"
        # print(path, os.path.join(out_path, out_file))
        # out_path = os.path.join(cfg.OUTPUT_DIR, os.path.split(vid)[0])

        if os.path.exists(out_file):
            print("{}. {} already exists".format(vid_no, os.path.split(out_file)[-1]))
            print("----------------------------------------------------------")
            continue
        if os.path.exists(os.path.join(cfg.DATA.TMP_FOLDER, vid_id)):
            print("{}. {} Decoded frames already exist".format(vid_no, vid_id))
            print("==========================================================")
            continue
        print("{}. Decoding {}...".format(vid_no, vid))
        # extract frames from video
        try:
            status, msg = decode_frames(cfg, vid_id)
            assert status, msg.decode('utf-8')

            print("{}. Processing {}...".format(vid_no, vid))
            dataset = VideoDataset(
                cfg, cfg.DATA.TMP_FOLDER, vid_id, num_frames
            )
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                sampler=None,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False,
            )

            # Perform multi-view test on the current video
            #v_pred_arr, n_pred_arr, feat_arr = perform_inference(test_loader, model, cfg)
            feat_arr = perform_inference(test_loader, model, cfg)
            print("{}. Finishing {}...".format(vid_no, vid))

            print("feats.shape", feat_arr.shape)
            
            np.savez(out_file, feats=feat_arr)

            # remove the extracted frames
            remove_frames(cfg, vid_id)
            print("Done.")
            print("----------------------------------------------------------")
        except Exception as e:
            print(e)
            print("Error in processing {}".format(vid))
            print("----------------------------------------------------------")
            if cfg.DATA.READ_VID_FILE:
                rejected_vids.append(vid)
            continue

    if cfg.DATA.READ_VID_FILE:
        print("Rejected Videos: {}".format(rejected_vids))

    print("----------------------------------------------------------")


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["configs/Kinetics/SLOWFAST_4x16_R50.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--vid_list",
        dest="vid_list",
        help="Path to the video list file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to the output directory",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--tmp_folder",
        dest="tmp_folder",
        help="Path to the temporary folder",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ckpt_path",
        dest="ckpt_path",
        help="Path to the checkpoint file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ckpt_type",
        dest="ckpt_type",
        help="Type of the checkpoint file caffe2 or pytorch",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)


    test(cfg)


if __name__ == "__main__":
    main()


#/media/abrar/Data_large/code/slowfast-feats$ python extract_feat.py --cfg configs/SLOWFAST_8x8_R50_jan4.yaml 
