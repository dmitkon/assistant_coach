from random import random
import pandas as pd
from copy import copy

# Пееремешать данные в векторах выборки
def shift_sample(sample, shifts):
    vectors = []
    for row in range(sample.shape[0]):
        for shift in shifts:
            vector = pd.DataFrame({'Part': [sample.iloc[row]['Part']]})
            
            for i, position in enumerate(shift):
                vector[f'Number_{i + 1}'] = sample.iloc[row][f'Number_{position}']
                vector[f'Ind_R_{i + 1}'] = sample.iloc[row][f'Ind_R_{position}']
                vector[f'Eff_R_{i + 1}'] = sample.iloc[row][f'Eff_R_{position}']

            vectors.append(vector)

    return pd.concat(vectors, ignore_index=True)

# Получить рандомную комбинацию для функции перемешивания
def get_random_shift(player_cnt):
    return [sorted(list(range(1, player_cnt + 1)), key=lambda A: random())]

# Получить циклические сдвиги для функции перемешивания
def get_cicle_shifts(player_cnt):
    shifts = []
    state = list(range(1, player_cnt + 1))
    
    for i, item in enumerate(state):
        state = state[-1:] + state[:-1]
        shifts.append(state)

    return shifts

# Добавить Признак индекс игрока (альтернатива номеру)
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
            indexes.append(players.get(sample.iloc[row][f'Number_{i + 1}']))

        new_sample.insert(2 + i*4, f'Index_{i + 1}', indexes)

    return new_sample

# Удалить признак
def drop_features(sample, features):
    return sample.drop(columns=features)

# Удалить номера игроков
def drop_numbers(sample, player_cnt):
    new_sample = copy(sample)
    
    for i in range(player_cnt):
        new_sample = drop_features(new_sample, [f'Number_{i + 1}'])
    
    return new_sample
