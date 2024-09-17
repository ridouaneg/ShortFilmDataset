import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='video_paths.csv', help="Path to the csv file with video paths")
    parser.add_argument("--input_dir", type=str, default='../data/faces', help="Path to the directory with individual face CSV files")
    parser.add_argument("--output_path", type=str, default='../data/faces.csv', help="Path to the output CSV file")
    return parser.parse_args()

def main(args):
    # Get parameters
    input_file = args.input_file
    input_dir = args.input_dir
    output_path = args.output_path

    # Get files
    files = list(Path(input_dir).glob('*.csv'))
    video_paths = pd.read_csv(input_file).video_path.tolist()
    video_ids = [Path(video_path).stem for video_path in video_paths]
    files = [file for file in files if file.stem in video_ids]

    logging.info(f"Found {len(files)} files")

    # Merge individual files into a single CSV file
    results = []
    for file in tqdm(files, desc="Merging faces", total=len(files)):
        try:
            df = pd.read_csv(file)
            results.append(df)
        except Exception as e:
            logging.error(f"Error occurred while reading faces from {file}: {str(e)}")
            continue
        
    results = pd.concat(results)
    results.to_csv(output_path, index=False)
    
    logging.info(f"Merged faces saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
