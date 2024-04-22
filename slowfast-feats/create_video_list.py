import os
import argparse
import glob
import subprocess

def probe_video(video_file):
    """
    A wrapper over ffprobe for extracting video meta info, including
    resolution (width, height), fps and number of frames
    """
    meta_info = {'width':-1, 'height':-1, "fps":-1, "num_frames":-1}
    status = False

    prob_cmd = ['ffprobe',
              '-v error',             # log to stderror
              '-of flat=s=_',         # output formating
              '-select_streams v:0',  # first video stream
              # query resolution, fps and # frames
              '-show_entries stream=width,height,r_frame_rate,nb_frames',
              "'{:s}'".format(video_file)]
    prob_cmd = ' '.join(prob_cmd)
    width, height, fps, num_frames = -1, -1, -1, -1

    prob_retry_cmd = ['ffprobe',
                    '-v error',             # log to stderror
                    '-count_frames',         # counting frames: slow
                    '-select_streams v:0',  # first video stream
                    # query # frames
                    '-show_entries stream=nb_read_frames',
                    '-of flat=s=_',         # output formating
                    "'{:s}'".format(video_file)]
    prob_retry_cmd = ' '.join(prob_retry_cmd)

    try:
        output = subprocess.check_output(prob_cmd, shell=True,
                                         stderr=subprocess.STDOUT)
        lines = output.decode("utf-8").split('\n')

        if len(lines) >=4:
            # getting resolution & fps
            str_width = lines[0].replace('streams_stream_0_width=', '')
            str_height = lines[1].replace('streams_stream_0_height=', '')
            str_fps = lines[2].replace('streams_stream_0_r_frame_rate=', '')
            str_fps = str_fps.replace('"', '')

            width = int(str_width)
            height = int(str_height)
            fps = float(str_fps.split('/')[0]) / float(str_fps.split('/')[1])

            if (width < 0) or (height < 0):
                return meta_info, status, "Could not probe video resolution!"

            meta_info['width'] = width
            meta_info['height'] = height
            meta_info['fps'] = fps

            # getting number of frames
            str_num_frames = lines[3].replace('streams_stream_0_nb_frames=', '')
            str_num_frames = str_num_frames.replace('"', '')

            if str_num_frames != 'N/A':
                num_frames = int(str_num_frames)
            else:
                # backup plan (if we could not probe # frames from the header)
                retry_output = subprocess.check_output(prob_retry_cmd, shell=True,
                                                       stderr=subprocess.STDOUT)
                lines = retry_output.decode("utf-8").split('\n')
                if len(lines) >= 1:
                    str_num_frames = lines[0].replace(
                        'streams_stream_0_nb_read_frames=', '')
                    str_num_frames = str_num_frames.replace('"', '')
                    num_frames = int(str_num_frames)

            meta_info['num_frames'] = num_frames

    except subprocess.CalledProcessError as err:
        return meta_info, status, err.output.decode("utf-8")

    status = True
    return meta_info, status, "Finished"

def main(video_dir, output_dir, ext):
    video_list = sorted(glob.glob(os.path.join(video_dir, "*.{:s}".format(ext))))
    output_file = os.path.join(output_dir, "patting_vid_list.csv")

    # prob the videos one by one
    frame_list = [None] * len(video_list)
    for idx, video_file in enumerate(video_list):
        meta_info, status, info = probe_video(video_file)

        # doesn't work?
        if status == False:
            print("Error processing {:s}".format(video_file))
            print(info)
            continue
        # sanity check
        """
        if meta_info['width'] != 720 or meta_info['height'] != 540\
           or meta_info['fps'] != 25.0 or meta_info['num_frames'] <= 0:
            print("{:s} has wrong meta info".format(video_file))
            print(meta_info)
            continue
        """
        # if meta_info['width'] != 1280 or meta_info['height'] != 720\
        #    or meta_info['fps'] != 30.0 or meta_info['num_frames'] <= 0:
        #     print("{:s} has wrong meta info".format(video_file))
        #     print(meta_info)


        frame_list[idx] = meta_info['num_frames']

    # write to file
    with open(output_file, 'w') as fid:
        for video_file, frame_num in zip(video_list, frame_list):
            if frame_num is not None:
                video_id = os.path.basename(video_file).replace(".{:s}".format(ext), '')
                fid.write("{:s} {:d}\n".format(video_id, frame_num))

    return

if __name__ == '__main__':
    description = 'Helper script for creating a video list from video files.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('video_dir', type=str,
                 help='Root folder with all video files.')
    p.add_argument('output_dir', type=str,
                 help='Output directory where csv file will be saved.')
    p.add_argument('-e', '--ext', type=str, default='mp4',
                 help='Video file name extension')
    main(**vars(p.parse_args()))
