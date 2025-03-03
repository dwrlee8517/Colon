from utils.dataset import RealColonDataset
import time
import matplotlib.pyplot as plt

data_dir = "/radraid/dongwoolee/real_colon_data"

def eval_skip(dataset:RealColonDataset):
    all_samples = dataset.samples
    sorted_all_samples = sorted(all_samples, key=lambda x: (x["video_id"], x["frame_id"]))
    # sort 
    skip_list = []
    prev_row = None
    for row in sorted_all_samples:
        if prev_row is None:
            prev_row = row
        
        prev_vid = prev_row["video_id"]
        prev_fid = prev_row["frame_id"]
        vid = row["video_id"]
        fid = row["frame_id"]
        if vid == prev_vid:
            skip = int(fid) - int(prev_fid) # 0 when prev_row does not exist
        else:
            skip = 0
        vid_fid = f"{vid}_{fid}"
        skip_list.append(skip)
            
        prev_row = row
    return skip_list

def plot_skip_histogram(skip_list, output_filename="radraid2/dongwoolee/Colon/data/skip_histogram.png"):
    
    # Create a new figure.
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram.
    # Here, we set bins automatically based on the min and max of skip_list.
    bins = range(min(skip_list), max(skip_list) + 2)  # +2 so that the last value is included.
    plt.hist(skip_list, bins=bins, edgecolor='black')
    
    # Add labels and title.
    plt.xlabel("Frame Skip (Difference in frame_id)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Frame Skips")
    
    # Save the plot to a file in the current directory.
    plt.savefig(output_filename)
    plt.close()
    print(f"Histogram saved to {output_filename}")

def main():
    start_time = time.time()
    dataset = RealColonDataset(data_dir, num_imgs=100000)
    skip_list = eval_skip(dataset)
    plot_skip_histogram(skip_list)
    print(skip_list)
    print(time.time() - start_time)


if __name__=='__main__':
    main()