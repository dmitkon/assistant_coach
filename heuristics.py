from random import random
import pandas as pd
from copy import copy

def shift_sample(sample, player_cnt):
    vectors = []
    for row in range(sample.shape[0]):
        permutation = sorted(list(range(1, player_cnt + 1)), key=lambda A: random())
        vector = pd.DataFrame({'Part': [sample.iloc[row]['Part']]})
        
        for i, position in enumerate(permutation):
            vector[f'Number_{i + 1}'] = sample.iloc[row][f'Number_{position}']
            vector[f'Ind_R_{i + 1}'] = sample.iloc[row][f'Ind_R_{position}']
            vector[f'Eff_R_{i + 1}'] = sample.iloc[row][f'Eff_R_{position}']

        vectors.append(vector)

    return pd.concat(vectors, ignore_index=True)

def add_player_index(sample, player_cnt):
    players = set()
    
    for i in range(player_cnt):
        players.update(set(sample[f'Number_{i + 1}']))

    players = sorted(list(players))
    players = dict(map(lambda a: (a[1], a[0]), zip(range(len(players)), players)))

    new_sample = copy(sample)

    for i in range(player_cnt):
        indexes = []
        
        for row in range(sample.shape[0]):
            indexes.append(players.get(sample[f'Number_{i + 1}'][row]))

        new_sample.insert(2 + i*4, f'Index_{i + 1}', indexes)

    return new_sample

def drop_features(sample, features):
    return sample.drop(columns=features)

def drop_numbers(sample, player_cnt):
    new_sample = copy(sample)
    
    for i in range(player_cnt):
        new_sample = drop_features(new_sample, [f'Number_{i + 1}'])
    
    return new_sample
