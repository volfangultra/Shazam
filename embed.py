#!/home/emin/Desktop/Projects/VjestackaInteligencija/Shazam/.venv/bin/ python3

import argparse
import os
import librosa
import numpy as np
import json
from help import *

SAVE_LIMIT = 3

parser = argparse.ArgumentParser(
                    prog='Embed audio files',
                    description='Takes all audio files and fingerprints them in a included json file')

parser.add_argument('input_folder')         
parser.add_argument('-output_file', default='data.json')
parser.add_argument('-neighborhood_size', default=51)
parser.add_argument('-embedding_type', default="Neighborhood", choices=["Neighborhood", "Maximum", "Heuristic"])
parser.add_argument('-maximum_chunk_size', default=5)
parser.add_argument('-heuristic_population_size', default=2000)
parser.add_argument('-heuristic_max_iter', default=100)
parser.add_argument('-heuristic_neighborhood', default=20)

args = parser.parse_args()

mp3_names = []
for root, dirs, files in os.walk(args.input_folder):
    for file in files:
        if file.endswith('.mp3'):
            mp3_names.append(file)



data = {}
if(os.path.exists(args.output_file)):
    with open(args.output_file, 'r') as json_file:
        data = json.load(json_file)

cnt = len(data)
for song in mp3_names[len(data):]:
    y, sr = librosa.load(os.path.join(args.input_folder, song))
    y = np.abs(librosa.stft(y))
    y_cords, x_cords = None, None
    if(args.embedding_type == "Neighborhood"):
        y_cords, x_cords = find_local_maxima(y, args.neighborhood_size)
    elif(args.embedding_type == "Maximum"):
        y_cords, x_cords = find_maxims(y, args.maximum_chunk_size)
    elif(args.embedding_type == "Heuristic"):
        y_cords, x_cords = find_local_maxima_heuristic(y, args.heuristic_population_size, args.heuristic_max_iter, args.heuristic_neighborhood)

    data[song] = {'y':y_cords, 'x':x_cords}
    cnt+=1

    if(cnt % SAVE_LIMIT == SAVE_LIMIT - 1):
        print(cnt, "files saved")
        with open(args.output_file, 'w') as outfile:
            json.dump(data, outfile)

with open(args.output_file, 'w') as outfile:
    json.dump(data, outfile)

