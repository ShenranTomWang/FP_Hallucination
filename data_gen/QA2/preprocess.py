import pandas as pd
import numpy as np
import os
import glob
import argparse

def find_and_rename_files(data_dir):
    """
    Find the QAQA files in the directory and rename evaluation file to test.csv
    
    Args:
        data_dir (str): Directory containing the QAQA files
        
    Returns:
        tuple: (adaptation_path, test_path) paths to the two files
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    adaptation_path = None
    evaluation_path = None
    
    # Look for the specific files
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).lower()
        
        if "adaptation" in filename:
            adaptation_path = csv_file
        elif "evaluation" in filename:
            evaluation_path = csv_file
    
    # If adaptation file not found by name, try to find it
    if not adaptation_path:
        # Check if there's exactly one CSV file (besides potentially the evaluation file)
        other_files = [f for f in csv_files if f != evaluation_path]
        if len(other_files) == 1:
            adaptation_path = other_files[0]
        else:
            raise FileNotFoundError(f"Could not find QAQA adaptation file in {data_dir}")
    
    # Rename evaluation file to test.csv if it exists
    test_path = None
    if evaluation_path:
        test_path = os.path.join(data_dir, "test.csv")
        if not os.path.exists(test_path):  # Only rename if test.csv doesn't already exist
            os.rename(evaluation_path, test_path)
            print(f"Renamed '{os.path.basename(evaluation_path)}' to 'test.csv'")
        else:
            print(f"'test.csv' already exists. Keeping both files.")
            test_path = evaluation_path  # Keep original path
    
    return adaptation_path, test_path

def split_dataset(adaptation_path, output_dir, train_ratio=0.8, random_seed=42):
    """
    Split the adaptation dataset into train and dev sets based on specified ratio.
    
    Args:
        adaptation_path (str): Path to the adaptation CSV file
        output_dir (str): Directory to save train and dev CSV files
        train_ratio (float): Proportion of data for training set (default: 0.8)
        random_seed (int): Random seed for reproducibility (default: 42)
    """
    
    # Read the CSV file
    print(f"Reading adaptation dataset from: {adaptation_path}")
    df = pd.read_csv(adaptation_path)
    
    # Calculate split sizes
    total_samples = len(df)
    train_size = int(total_samples * train_ratio)
    dev_size = total_samples - train_size
    
    print(f"Total adaptation samples: {total_samples}")
    print(f"Train samples: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Dev samples: {dev_size} ({(1-train_ratio)*100:.1f}%)")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create train and dev indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:]
    
    # Split the dataframe
    train_df = df.iloc[train_indices].reset_index(drop=True)
    dev_df = df.iloc[dev_indices].reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV files
    train_path = os.path.join(output_dir, "train.csv")
    dev_path = os.path.join(output_dir, "dev.csv")
    
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    
    print(f"\nTrain set saved to: {train_path}")
    print(f"Dev set saved to: {dev_path}")
    
    # Optional: Print a summary of the split
    print("\n" + "="*50)
    print("Split Summary:")
    print("="*50)
    print(f"Train set shape: {train_df.shape}")
    print(f"Dev set shape: {dev_df.shape}")
    
    # Check distribution of 'has_invalid' vs 'all_valid' in both sets
    if 'all_assumptions_valid' in df.columns:
        print("\nDistribution of assumption validity:")
        print("-" * 30)
        train_dist = train_df['all_assumptions_valid'].value_counts()
        dev_dist = dev_df['all_assumptions_valid'].value_counts()
        
        print("Train set:")
        for val, count in train_dist.items():
            print(f"  {val}: {count} ({count/len(train_df)*100:.1f}%)")
        
        print("\nDev set:")
        for val, count in dev_dist.items():
            print(f"  {val}: {count} ({count/len(dev_df)*100:.1f}%)")
    
    return train_df, dev_df

def main():
    parser = argparse.ArgumentParser(description='Split QAQA dataset into train, dev, and test sets')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory containing QAQA adaptation and evaluation CSV files'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save train and dev CSV files (default: same as data_dir)'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Proportion of adaptation data for training set (default: 0.8)'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set output directory to data_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    # Validate train ratio
    if not 0 < args.train_ratio < 1:
        print("Error: train_ratio must be between 0 and 1 (exclusive)")
        return
    
    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' not found")
        return
    
    try:
        # Find and rename files
        adaptation_path, test_path = find_and_rename_files(args.data_dir)
        
        # Report about test file
        if test_path:
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                print(f"\nTest set found: {os.path.basename(test_path)}")
                print(f"Test samples: {len(test_df)}")
        
        # Split adaptation dataset
        print("\n" + "="*50)
        print("Splitting adaptation dataset:")
        print("="*50)
        train_df, dev_df = split_dataset(
            adaptation_path=adaptation_path,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            random_seed=args.random_seed
        )
        
        print("\n" + "="*50)
        print("Processing complete!")
        print("="*50)
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nFiles created/updated:")
        print(f"  - {os.path.join(args.output_dir, 'train.csv')}")
        print(f"  - {os.path.join(args.output_dir, 'dev.csv')}")
        if test_path and os.path.exists(test_path):
            print(f"  - {test_path} (renamed from evaluation file)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nLooking for files with names containing 'adaptation' and 'evaluation'")
        print(f"Files found in {args.data_dir}:")
        for f in glob.glob(os.path.join(args.data_dir, "*")):
            print(f"  - {os.path.basename(f)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()