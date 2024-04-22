#!/usr/bin/env python3

import slowfast.config.defaults as defcfg
import slowfast.utils.checkpoint as cu


# -----------------------------------------------------------------------------
# Additional Data options
# -----------------------------------------------------------------------------

# stride in frames for feature extraction
defcfg._C.DATA.STRIDE = 16

# Flag to set video file/image file processing
defcfg._C.DATA.READ_VID_FILE = True

# File extension of video files
defcfg._C.DATA.VID_FILE_EXT = ".mp4"

# Sampling width / height of each frame
defcfg._C.DATA.SAMPLE_SIZE = [400, 300]

# the video list file
defcfg._C.DATA.VID_LIST = ""

# the working folder where we will unpack the video frames
defcfg._C.DATA.TMP_FOLDER = ""
defcfg._C.DATA.FPS = 25
defcfg._C.BN.FREEZE: True
defcfg._C.DATA.CROP_SIZE = 224

def get_cfg():
    """
    Get a copy of the default config.
    """
    return defcfg.assert_and_infer_cfg(defcfg._C.clone())


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    print(args)
    # Load config from cfg.
    if args.cfg_files is not None:
        cfg.merge_from_file(args.cfg_files[0])
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed

    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.ckpt_path is not None:
        cfg.TRAIN.CHECKPOINT_FILE_PATH = args.ckpt_path
    
    if args.ckpt_type is not None:
        cfg.TRAIN.CHECKPOINT_TYPE = args.ckpt_type
    
    if args.tmp_folder is not None:
        cfg.DATA.TMP_FOLDER = args.tmp_folder

    if args.vid_list is not None:
        cfg.DATA.VID_LIST = args.vid_list
    
    # Create the checkpoint dir.
    # cu.make_checkpoint_dir(cfg.OUTPUT_DIR)

    return cfg
