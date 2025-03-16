import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os

data_basepath = "/radraid/dongwoolee/real_colon_data"

def get_frame(basepath, video_id, frame_id):
    frames_dir = os.path.join(basepath, (video_id + "_frames"))
    frame_path = os.path.join(frames_dir, (video_id + "_" + frame_id + ".jpg"))
    return frame_path

def get_annotation(basepath, video_id, frame_id):
    annot_dir = os.path.join(basepath, (video_id + "_annotations"))
    annot_path = os.path.join(annot_dir, (video_id + "_" + frame_id + ".xml"))
    return annot_path

def get_size(annotpath):
    try:
        tree = ET.parse(annotpath)
        size_elem = tree.find("size")
        if size_elem is not None:
            # Build a tuple of integer values from each child element.
            return tuple(int(child.text) for child in size_elem)
    except Exception as e:
        print(f"Error parsing {annotpath}: {e}")
    return None

def process_row(row, data_basepath):
    """
    Given a row (with video_id and frame_id), compute the image path, annotation path, 
    and then parse the XML to extract the size.
    """
    video_id = str(row.video_id)
    frame_id = str(row.frame_id)
    imgpath = get_frame(data_basepath, video_id, frame_id)
    annotpath = get_annotation(data_basepath, video_id, frame_id)
    size = get_size(annotpath)
    return (imgpath, size)

def process_chunk(chunk, data_basepath, max_workers=8):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a named tuple by setting name="Row"
        futures = executor.map(lambda row: process_row(row, data_basepath),
                                 chunk.itertuples(index=False, name="Row"))
        results = list(tqdm(futures, total=len(chunk), desc="Processing Chunk"))
    # Filter out rows with missing data.
    chunk_results = [(imgpath, size) for imgpath, size in results if imgpath is not None and size is not None]
    return chunk_results

def process_all_frames(data_basepath, frames_polyps, chunk_size=100000, max_workers=8, output_csv="img_sizes.csv"):
    # Filter only frames where is_polyps_frame is 1.
    filtered_frames = frames_polyps[frames_polyps["is_polyps_frame"] == 0]
    
    total_rows = len(filtered_frames)
    chunks = [filtered_frames.iloc[i:i+chunk_size] for i in range(0, total_rows, chunk_size)]
    
    all_results = []
    for i, chunk in enumerate(chunks):
        chunk_results = process_chunk(chunk, data_basepath, max_workers=max_workers)
        all_results.extend(chunk_results)
        print(f"Processed chunk {i+1}/{len(chunks)}")
    
    # Combine all results into a single DataFrame.
    df_final = pd.DataFrame(all_results, columns=['ImagePath', 'Size'])
    # Save the entire DataFrame to CSV at once.
    df_final.to_csv(output_csv, index=False)
    print(f"Entire dataset saved to {output_csv}")
    return df_final


if __name__ == '__main__':
    # Define your data folder where frames_polyps.csv is located.
    datafolder = "/radraid2/dongwoolee/Colon/data"
    frames_polyps = pd.read_csv(os.path.join(datafolder, "frames_polyps.csv"))
    
    # Process all frames in chunks and combine the results, then save to a single CSV.
    df_img_sizes = process_all_frames(data_basepath, frames_polyps, chunk_size=10000, max_workers=8, output_csv="img_sizes.csv")
    print(df_img_sizes.head())