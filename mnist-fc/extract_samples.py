"""
Extract a few images from the MNIST dataset and save as raw 28x28 byte files.
Usage: python extract_samples.py [--output-dir DIR] [--count N]
"""
import os
import argparse
import struct
import numpy as np
from torchvision import datasets

def main():
    parser = argparse.ArgumentParser(description='Extract MNIST samples as .raw files')
    parser.add_argument('--output-dir', type=str, default='samples', help='directory to save .raw files')
    parser.add_argument('--count', type=int, default=5, help='number of samples to extract')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MNIST test set (no normalisation, just raw pixels)
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    for i in range(args.count):
        img, label = dataset[i]
        # img is PIL Image, convert to numpy uint8
        arr = np.array(img, dtype=np.uint8)
        # Save as raw bytes (784 bytes)
        filename = os.path.join(args.output_dir, f'sample_{i}_label{label}.raw')
        with open(filename, 'wb') as f:
            f.write(arr.tobytes())
        print(f'Saved {filename} (label: {label})')

if __name__ == '__main__':
    main()
