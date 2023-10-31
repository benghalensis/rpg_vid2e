import argparse
from operator import sub
import os
import esim_torch
import numpy as np
import glob
import cv2
import tqdm
import torch

class EventStorer():
    def __init__(self, freq_mult, save_path):
        self.freq_mult = freq_mult
        self.save_path = save_path

        self.event_counter = 0
        self.save_counter = 0
        self.accumulated_events = np.zeros(shape=(1,4), dtype=np.int64)

    def append(self, sub_events):
        self.accumulated_events = np.concatenate((self.accumulated_events, np.array([sub_events['t'].numpy(), 
                                                                                     sub_events['x'].numpy(), 
                                                                                     sub_events['y'].numpy(), 
                                                                                     sub_events['p'].numpy()]).T))
        self.event_counter += 1

        if self.event_counter % self.freq_mult == 0:
            self.save()

    def save(self):
        np.savez_compressed(os.path.join(self.save_path, "%06d_event_cam.npz" % self.save_counter), event_data=self.accumulated_events[1:])
        self.accumulated_events = np.zeros(shape=(1,4), dtype=np.int64)
        self.save_counter += 1

def process(args, image_files, timestamp_ns, output_dir): 
    # Initialize the ESIM event generator
    esim = esim_torch.ESIM(args.contrast_threshold_negative,
                        args.contrast_threshold_positive,
                        args.refractory_period_ns)
    
    # Logging
    pbar = tqdm.tqdm(total=len(image_files)-1)
    num_events = 0
    counter = 0

    # Initialize the event storer
    event_storer = EventStorer(args.freq_mult, output_dir)

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
            cv2.imwrite(os.path.join(output_dir, os.path.join("event_visualization", "%010d.png" % counter)), image_color)
 
        # Accumulate and save events
        event_storer.append(sub_events)

        # Update the logger
        pbar.set_description(f"Num events generated: {num_events}")
        pbar.update(1)
        counter += 1

    # Save the remaining events
    event_storer.save()

def parse_events(args):
    event_files = sorted(glob.glob(os.path.join(args.output_dir, "*.npz")))

    event_data = []
    for event in tqdm.tqdm(event_files):
        data = np.load(event)
        event_data.append(np.array([data['t'], data['x'], data['y'], data['p']]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--contrast_threshold_negative", "-cn", type=float, default=0.2)
    parser.add_argument("--contrast_threshold_positive", "-cp", type=float, default=0.2)
    parser.add_argument("--refractory_period_ns", "-rp", type=int, default=500000)
    parser.add_argument("--image_fps",  default=1000, type=int)
    parser.add_argument("--task",  default="gen")
    parser.add_argument("--skip_one_in_every",  default=0, type=int)
    parser.add_argument("--imu_file_path",  default="", type=str)
    parser.add_argument("--image_directory",  default="", type=str)
    parser.add_argument("--output_dir", "-o", default="")
    parser.add_argument("--visualize_events", action="store_true")

    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--data_folder_name", type=str, default="")
    parser.add_argument("--freq_mult", type=int, default=100)
    
    args = parser.parse_args()

    if args.task == "gen_simple":
        # Create output directory 
        print(f"Processing folder {args.image_directory}... Generating events in {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

        if args.visualize_events:
            os.makedirs(os.path.join(args.output_dir, "event_visualization"), exist_ok=True)       # constructor

        timestamps = np.genfromtxt(args.imu_file_path, dtype="float64")
        timestamps_ns = (timestamps * 1e9).astype("int64")
        timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

        image_files = sorted(glob.glob(os.path.join(args.image_directory, "*.png")))

        print(f"Generating events with cn={args.contrast_threshold_negative}, cp={args.contrast_threshold_positive} and rp={args.refractory_period_ns}")
        process(args, image_files, timestamps_ns, args.output_dir)

    if args.task == "gen_tartanair":
        dataset_dir = args.dataset_dir # /mnt/e/EventCamera
        env_name = args.env_name
        data_folder_name = args.data_folder_name

        # Find the folders that start with P, like P000, P001
        traj_folder_base_path = os.path.join(os.path.join(dataset_dir, env_name), data_folder_name)
        traj_folder_list = [os.path.join(traj_folder_base_path, d) for d in os.listdir(traj_folder_base_path) if os.path.isdir(os.path.join(traj_folder_base_path, d)) and d.startswith('P')]
        traj_folder_list.sort()
        
        for traj_folder in traj_folder_list:
            print ('*** {} ***'.format(traj_folder))
            images_directory = os.path.join(traj_folder, "image_lcam_front")
            image_files = sorted(glob.glob(os.path.join(images_directory, "*.png")))
            imu_time_filepath = os.path.join(traj_folder, os.path.join("events", "hf_time_pose_lcam_front.txt"))
            events_output_directory = os.path.join(traj_folder, os.path.join("events", "events_output"))

            os.makedirs(events_output_directory, exist_ok=True)

            # IMU timestamps
            timestamps = np.genfromtxt(imu_time_filepath, dtype="float64")
            timestamps_ns = (timestamps * 1e9).astype("int64")
            timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

            # Output directory
            process(args, image_files, timestamps_ns, events_output_directory)


    if args.task == "parse":
        parse_events(args)
