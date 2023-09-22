import argparse
from operator import sub
import os
import esim_torch
import numpy as np
import glob
import cv2
import tqdm
import torch

def process_dir(args):
    print(f"Processing folder {args.image_directory}... Generating events in {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize_events:
        os.makedirs(os.path.join(args.output_dir, "event_visualization"), exist_ok=True)

    # constructor
    esim = esim_torch.ESIM(args.contrast_threshold_negative,
                           args.contrast_threshold_positive,
                           args.refractory_period_ns)

    timestamps = np.genfromtxt(args.imu_file_path, dtype="float64")
    timestamps_ns = (timestamps * 1e9).astype("int64")
    timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

    image_files = sorted(glob.glob(os.path.join(args.image_directory, "*.png")))
    
    pbar = tqdm.tqdm(total=len(image_files)-1)
    num_events = 0

    counter = 0
    for image_file, timestamp_ns in zip(image_files, timestamps_ns):
        if (counter % (args.skip_one_in_every + 1) != 0):
            counter += 1
            continue
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()

        sub_events = esim.forward(log_image, timestamp_ns)

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is None:
            continue

        sub_events = {k: v.cpu() for k, v in sub_events.items()}    
        num_events += len(sub_events['t'])

        if args.visualize_events:
            image_color = np.stack([image,image,image],-1)
            image_color[sub_events['y'], sub_events['x'], :] = 0
            image_color[sub_events['y'], sub_events['x'], sub_events['p']] = 255
            cv2.imwrite(os.path.join(args.output_dir, os.path.join("event_visualization", "%010d.png" % counter)), image_color)
 
        # do something with the events
        np.savez(os.path.join(args.output_dir, "%010d.npz" % counter), **sub_events)
        pbar.set_description(f"Num events generated: {num_events}")
        pbar.update(1)
        counter += 1

def parse_events(args):
    event_files = sorted(glob.glob(os.path.join(args.output_dir, "*.npz")))

    event_data = []
    for event in tqdm.tqdm(event_files):
        data = np.load(event)
        event_data.append(np.array([data['t'], data['x'], data['y'], data['p']]))

    # import pandas as pd
    # pd.to_csv(os.path.join(args.output_dir, "events.txt"), sep=' ', header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--contrast_threshold_negative", "-cn", type=float, default=0.2)
    parser.add_argument("--contrast_threshold_positive", "-cp", type=float, default=0.2)
    parser.add_argument("--refractory_period_ns", "-rp", type=int, default=500000)
    parser.add_argument("--image_fps",  default=1000, type=int)
    parser.add_argument("--task",  default="gen")
    parser.add_argument("--skip_one_in_every",  default=0, type=int)
    parser.add_argument("--imu_file_path",  default="", type=str, required=True)
    parser.add_argument("--image_directory",  default="", type=str, required=True)
    parser.add_argument("--output_dir", "-o", default="", required=True)
    parser.add_argument("--visualize_events", action="store_true")
    
    args = parser.parse_args()


    if args.task == "gen":
        print(f"Generating events with cn={args.contrast_threshold_negative}, cp={args.contrast_threshold_positive} and rp={args.refractory_period_ns}")
        process_dir(args)

    if args.task == "parse":
        parse_events(args)
