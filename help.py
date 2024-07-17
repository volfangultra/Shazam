import numpy as np
from collections import Counter

def find_local_maxima(matrix, neighborhood_size):
    if neighborhood_size % 2 == 0:
        raise ValueError("Neighborhood size must be odd.")
    
    pad_size = neighborhood_size // 2
    y_cords = []
    x_cords = []

    for i in range(pad_size, matrix.shape[0] - pad_size):
        for j in range(pad_size, matrix.shape[1] - pad_size):
            neighborhood = matrix[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            center_value = matrix[i, j]
            
            temp = neighborhood[neighborhood != center_value]
            if len(temp) != 0 and center_value > np.max(temp):
                y_cords.append(i)
                x_cords.append(j)
    
    return y_cords, x_cords

def find_maxims(matrix, chunk_size):
    y_cords = list(np.argmax(matrix, axis=0))
    x_cords = list(range(len(y_cords)))

    y_rez, x_rez = [], []
    for i in range(0, len(y_cords), chunk_size):
        # Get the current chunk
        chunk = y_cords[i:i + chunk_size]

        max_index = np.argmax(chunk)

        global_max_index = i + max_index

        y_rez.append(int(y_cords[global_max_index]))
        x_rez.append(int(x_cords[global_max_index]))

    return y_rez, x_rez

def find_local_maxima_heuristic(matrix, population_size, num_iterations, neighborhood_size):
    m, n = matrix.shape
    y_coords = np.random.uniform(neighborhood_size, m, population_size).astype(int)
    x_coords = np.random.uniform(neighborhood_size, n, population_size).astype(int)
    y_rez = []
    x_rez = []
    matrix = np.pad(matrix, pad_width=neighborhood_size, mode='constant', constant_values=-np.inf)
    for y, x in zip(y_coords, x_coords):
        for _ in range(num_iterations):
            okolina = matrix[y-neighborhood_size:y+neighborhood_size + 1 , x-neighborhood_size:x+neighborhood_size + 1]
            if okolina.size == 0:
                break
            max_index = np.argmax(okolina)
            dim = (2*neighborhood_size + 1)
            if(max_index // dim == neighborhood_size and max_index % dim == neighborhood_size):
                break
            y = y + max_index // dim - neighborhood_size
            x = x + max_index % dim - neighborhood_size

        if(y - neighborhood_size not in y_rez or x - neighborhood_size not in x_rez):
            y_rez.append(int(y - neighborhood_size))
            x_rez.append(int(x - neighborhood_size))
    
    return y_rez, x_rez

def evaluate_similarity(x_original, y_original, x_test, y_test, tolerance_y, tolerance_x):
    hist = []
    for i in range(len(x_original)):
        for j in range(len(x_test)):
            if(abs(y_original[i] - y_test[j]) < tolerance_y):
                hist.append(x_original[i] - x_test[j])

    hist = Counter(hist)
    if(len(hist) == 0):
        return 0
    time, nums = hist.most_common(1)[0]
    rez = nums
    for i in range(1, tolerance_x):
        if(time - i > 0):
            rez += hist[time - i]
        if(time + i < len(hist)):
            rez += hist[time + i]

    return rez