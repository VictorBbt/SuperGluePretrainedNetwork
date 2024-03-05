import match_pairs as mp
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import argparse
import time
########### HELPERS ############

def create_folder(path):
    try:
        # Use makedirs to create the folder and its parent directories if they don't exist
        os.makedirs(path)
        print(f"Folder created at {path}")
    except FileExistsError:
        print(f"Folder already exists at {path}")


def create_text_file(file_path, input_name, subset_names):
    try:
        with open(file_path, 'w') as file:
            for sub in subset_names:
                file.write(f'{input_name} {sub}\n')
        print(f"{file_path} text file created at {file_path}")
    except IOError:
        print(f"Error creating the text file at {file_path}")

############ BENCHMARKING #############
        
def match_keypoints_csv(data_dir, data_df, output_dir = f'./data/streets/results', debug = False):


    superglue_mode = 'outdoor'
    metrics = 'nb_keypoints'

    dict_results = {}
    data_length = len(data_df)
    nb_img_to_treat = 10
    treated_img = 0
    treated_lines = 0

    if debug:
        print(f'Find keypoints with mode {superglue_mode}, metrics {metrics}')
        print(f'Will treat the first {nb_img_to_treat} points or the full data of length {data_length}')

    # update subset fiename for each image and then perform comparison
    while treated_img <= nb_img_to_treat and treated_lines < data_length:
        

        current_img = str(data_df.iloc[treated_lines][0])
        subset_filenames = []

        while str(data_df.iloc[treated_lines][0]) == current_img and treated_lines < data_length:
            subset_filenames.append(str(data_df.iloc[treated_lines][1]))
            treated_lines += 1
        
        if(debug):
            print('___________________________________________')
            print(f'Comparison n{treated_img}: {current_img}')
            print(f'Found {len(subset_filenames)} img to compare')

        input_img_filename = current_img + '.jpg'
        subset_filenames = [f + '.jpg' for f in subset_filenames]

        if(debug):
            print('Creating folders...')
        res_path = f'{output_dir}/matches_{input_img_filename[:-4]}'
        create_folder(res_path)
        pairs_path = f'{res_path}/pairs.txt'
        create_text_file(pairs_path, input_img_filename, subset_filenames)

        if(debug):
            print('Running Superglue...')
            
        os.system(f'python3 match_pairs.py --viz --superglue {superglue_mode} --input_pairs {pairs_path} --input_dir {data_dir} --output_dir {res_path} --resize -1')

        list_results_current_img = []
        for img in subset_filenames:
            match_filename = f'{input_img_filename[:-4]}_{img[:-4]}_matches.npz'
            match_result_path = res_path + match_filename
            npz_res = np.load(match_result_path)
            if metrics=='nb_keypoints':
                list_results_current_img.append(len(npz_res["matches"]) - list(npz_res["matches"]).count(-1))

        dict_results[current_img] = list_results_current_img

        if(debug):
            print(f'Found mean nb of keypoints: {np.mean(list_results_current_img)}')
        
        treated_img +=1
    return dict_results

############ VISUALIZATION ##############

def save_results(dict_results, output_folder,  threshold=100):

    all_kps = []
    mean_matches = []
    max_kps = []
    for l in dict_results.values():
        all_kps+= l
        mean_matches.append(np.mean(l))
        max_kps.append(np.max(l))
    # Plotting the histogram
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    axes = axes.flatten()
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    ax1.hist(all_kps, bins=100, color='blue', edgecolor='black')
    ax1.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    ax1.set_xlabel('Keypoints')
    ax1.set_ylabel('Nb of appearances')
    ax1.set_title('Nb keypoints')

    ax2.hist(mean_matches, bins=50, color='green', edgecolor='black')
    ax2.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    ax2.set_xlabel('Mean Keypoints for an image throughout all comparisons')
    ax2.set_ylabel('Nb of apperances')
    ax2.set_title('Mean of keypoints')

    ax3.hist(max_kps, bins=50, color='green', edgecolor='black')
    ax3.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    ax3.set_xlabel('Max Keypoints for an image throughout all comparisons')
    ax3.set_ylabel('Nb of apperances')
    ax3.set_title('Maxkeypoints')

    # Display the plot
    plt.savefig(output_folder)


############### MAIN FUNCTION #############
    
img_folder_default = "./data/streets/data-2/street_view_images_raw"
output_folder_default = "./data/streets/results"


parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--img_folder", required=True, default= "./data/streets/data-2/street_view_images_raw",
                    help="Name of the folder containing the images.")
parser.add_argument("--csv_file", required=True,
                    help="CSV files containing pairs to compare")
parser.add_argument("--output_folder", required = True, default="./data/streets/results",
                        help="Folder where results will be saved (create folder if not exists)")
parser.add_argument("--threshold", default=100,
                        help="Threshold that will be plotted on the results, and later for classification")

args = parser.parse_args()

start_time = time.time()
data_df = pd.read_csv(args.csv_file, sep=',')
dict_results = match_keypoints_csv(args.img_folder, data_df, output_dir = args.output_dir, debug = False)
df_results = pd.DataFrame.from_dict(dict_results)
df_results.to_csv(args.output_folder)
save_results(dict_results=dict_results, output_folder=args.output_folder, threshold = args.threshold)

print('___________________________________')
print(f'Finished execution in {time.time() - start_time} seconds')
print(f'Results saved at {args.output_folder}')