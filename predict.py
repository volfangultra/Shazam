import argparse
import librosa
import numpy as np
import os
import json
from help import *

parser = argparse.ArgumentParser(
                    prog='Predict the name of the song',
                    description='Takes a chunk of a song in mp3 format and outputs the name')

parser.add_argument('input_file')       
parser.add_argument('-dataset', default='data.json')
parser.add_argument('-tolerance_x', default=25)
parser.add_argument('-tolerance_y', default=25)
parser.add_argument('-neighborhood_size', default=69)
parser.add_argument('-embedding_type', default="Neighborhood", choices=["Neighborhood", "Maximum", "Heuristic"])
parser.add_argument('-maximum_chunk_size', default=10)
parser.add_argument('-heuristic_population_size', default=2000)
parser.add_argument('-heuristic_max_iter', default=100)
parser.add_argument('-heuristic_neighborhood', default=5)

args = parser.parse_args()
args.heuristic_population_size = int(args.heuristic_population_size)
args.heuristic_max_iter, args.heuristic_neighborhood = int(args.heuristic_max_iter), int(args.heuristic_neighborhood)
args.neighborhood_size = int(args.neighborhood_size)

if(not os.path.exists(args.dataset)):
    raise "FILE DOESN'T EXIST"

with open(args.dataset, 'r') as json_file:
    data = json.load(json_file)
    y, sr = librosa.load(args.input_file)
    print("FILE LOADED")
    y = np.abs(librosa.stft(y))
    y_cords, x_cords = None, None
    if(args.embedding_type == "Neighborhood"):
        y_cords, x_cords = find_local_maxima(y, args.neighborhood_size)
    elif(args.embedding_type == "Maximum"):
        y_cords, x_cords = find_maxims(y, args.maximum_chunk_size)
    elif(args.embedding_type == "Heuristic"):
        y_cords, x_cords = find_local_maxima_heuristic(y, args.heuristic_population_size, args.heuristic_max_iter, args.heuristic_neighborhood)
    print("FINGERPRINTING DONE")

    most_similar = 0
    best_song = ""
    for song, values in data.items():
        similarity = evaluate_similarity(values['x'], values['y'], x_cords, y_cords, args.tolerance_y, args.tolerance_x)
        print(song, similarity)
        if(similarity > most_similar):
            best_song = song
            most_similar = similarity
    
    print("FOUND SONG:", best_song)