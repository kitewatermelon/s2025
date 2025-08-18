import os
import SimpleITK as sitk
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

def visualize_slices(output_root, num_examples=3):
    """
    Visualize a few random examples of the extracted 2D slices.
    
    Args:
        output_root (str): Root directory of the output dataset
        num_examples (int): Number of examples to visualize
    """
    mha_files = []
    for dirpath, dirnames, filenames in os.walk(output_root):
        for filename in filenames:
            if filename.endswith('.mha'):
                mha_files.append(os.path.join(dirpath, filename))
    
    if not mha_files:
        print("No MHA files found in the output directory.")
        return
    
    num_to_visualize = min(num_examples, len(mha_files))
    selected_files = random.sample(mha_files, num_to_visualize)
    
    fig, axes = plt.subplots(1, num_to_visualize, figsize=(5*num_to_visualize, 5))
    if num_to_visualize == 1:
        axes = [axes]
    
    for i, file_path in enumerate(selected_files):
        image = sitk.ReadImage(file_path)
        
        array = sitk.GetArrayFromImage(image)

        if array.ndim > 2:
            array = array[0]
        
        axes[i].imshow(array, cmap='gray')
        axes[i].set_title(os.path.basename(os.path.dirname(file_path)) + '/' + os.path.basename(file_path))
        axes[i].axis('off')
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_root, 'visualization.png')
    plt.savefig(viz_path)
    print(f"Visualization saved to {viz_path}")
    
    plt.show()


# def extract_middle_slice(input_path, output_path):
#     """
#     Extract the middle axial slice from a 3D MHA file and save it as a 2D MHA file.
    
#     Args:
#         input_path (str): Path to the input 3D MHA file
#         output_path (str): Path to save the output 2D MHA file
#     """
#     image_3d = sitk.ReadImage(input_path)
    
#     size = image_3d.GetSize()
    
#     middle_slice_index = size[2] // 2

#     extract_filter = sitk.ExtractImageFilter()
    
#     new_size = list(size)
#     new_size[2] = 0
    
#     start = [0, 0, middle_slice_index]
    
#     extract_filter.SetSize(new_size)
#     extract_filter.SetIndex(start)
    
#     image_2d = extract_filter.Execute(image_3d)
    
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     sitk.WriteImage(image_2d, output_path)
    
#     print(f"Extracted middle slice from {input_path} to {output_path}")

import SimpleITK as sitk
import os

import SimpleITK as sitk
import os

def extract_middle_slice(input_path, output_path, expand=0):
    """
    Extract the middle axial slice from a 3D MHA file and save it as a 2D (or pseudo-2.5D) MHA file.
    
    Args:
        input_path (str): Path to the input 3D MHA file
        output_path (str): Path to save the output 2D/2.5D MHA file
        expand (int): Number of slices to include above and below the middle slice for 2.5D. 0 means just middle slice.
    """
    # Force expand=0 if input is CT
    # if "ct" in input_path.lower() and "cbct" not in input_path.lower():
    #     expand = 0

    image_3d = sitk.ReadImage(input_path)
    size = image_3d.GetSize()  # (x, y, z)
    mid_index = size[2] // 2

    start_index = max(0, mid_index - expand)
    end_index = min(size[2] - 1, mid_index + expand)

    slices = []
    for idx in range(start_index, end_index + 1):
        extract_filter = sitk.ExtractImageFilter()
        extract_filter.SetSize([size[0], size[1], 0])
        extract_filter.SetIndex([0, 0, idx])
        slice_2d = extract_filter.Execute(image_3d)
        slices.append(slice_2d)

    if len(slices) == 1:
        output_image = slices[0]
    else:
        output_image = sitk.JoinSeries(slices)
        # output_image = sitk.PermuteAxes(output_image, [2, 0, 1])  # [slice, x, y] -> [x, y, slice]
        output_image = sitk.Cast(output_image, sitk.sitkFloat32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(output_image, output_path)

    print(f"Saved {'2.5D' if expand > 0 else '2D'} slice(s) from {input_path} to {output_path}")

def process_dataset(input_root, output_root, expand):
    """
    Process all MHA files in the dataset directory structure.
    
    Args:
        input_root (str): Root directory of the input dataset
        output_root (str): Root directory where to save the output 2D slices
    """
    os.makedirs(output_root, exist_ok=True)
    
    for dirpath, dirnames, filenames in os.walk(input_root):
        relpath = os.path.relpath(dirpath, input_root)
        
        if relpath != '.':
            output_dir = os.path.join(output_root, relpath)
            os.makedirs(output_dir, exist_ok=True)
        
        for filename in filenames:
            if filename.endswith('.mha'):
                input_file = os.path.join(dirpath, filename)
                output_file = os.path.join(output_root, relpath, filename)
                
                try:
                    extract_middle_slice(input_file, output_file, expand)
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extract middle axial slices from 3D MHA files.')
    parser.add_argument('--input', type=str, required=True, help='Input root directory')
    parser.add_argument('--output', type=str, required=True, help='Output root directory')
    parser.add_argument('--expand', type=int, default=2, help='2.5D slice')
    parser.add_argument('--visualize', action='store_true', help='Visualize some extracted slices after processing')
    parser.add_argument('--num-examples', type=int, default=3, help='Number of examples to visualize')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.expand)
    
    if args.visualize:
        visualize_slices(args.output, args.num_examples)
    
if __name__ == "__main__":
    main()
