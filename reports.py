from decimal import Decimal
import pandas as pd
import numpy as np
from copy import copy
from functools import reduce
import coach

# Прочитать данные из файлов отчётов
def read_reports(matches, parts, path):
    reports = []

    for match in range(matches):
        for part in range(parts):
            data = pd.read_excel(path + 'M' + str(match + 1) + 'P' + str(part + 1) + '.xls')
            data.rename(columns=dict(zip(list(data), list(data.iloc[3]))), inplace=True)
            data = data[4:data.shape[0] - 1]
            data['Part'] = [part + 1 for i in range(data.shape[0])]
            reports.append(data)

    return reports

# Вызвать функции для каждого отчёта
def apply_function_data(f, reports):
    return list(map(lambda report: f(report), reports))

# Получить отчёты с разбиением строки игрока на номер и фамилию
def get_split_players(reports):
    return apply_function_data(split_players, reports)

# Разбить строки игрока на номер и фамилию
def split_players(data):
    new_data = copy(data)
    numbers = []
    last_names = []

    for player in new_data['Player']:
        if player != "Team":
            numbers.append(int(player.split()[0]))
            last_names.append(player.split()[1])
        else:
            numbers.append(0)
            last_names.append(player)

    new_data['Number'] = numbers
    new_data['Last_name'] = last_names

    return new_data.drop(columns=['Player'])

# Получить отчёты с заполненными пропусками в столбце с обобщениями
def get_fill_general_nan(keys, reports):
    new_reports = copy(reports)

    for key in keys:
        new_reports = apply_function_data(lambda data: fill_general_nan(key, data), new_reports)
    
    return new_reports

# Заполнить пропуски в столбце с обобщениями
def fill_general_nan(key, data):
    new_data = copy(data)
    item = ""
    items = []

    for elem in new_data[key]:
        if type(elem) == str:
            item = elem

        items.append(item)

    new_data[key] = items

    return new_data

# Получить отчёты только с выбранными колонками
def get_by_keys(keys, reports):
    return apply_function_data(lambda data: data[keys], reports)

# Получить отчёты с отфильтрованными по значениям строками
def filter_by_values(data, filters):
    new_data = copy(data)

    for key in filters:
        new_data = new_data[new_data[key].isin(filters.get(key))]

    return new_data

# Исмключить данные по команде
def get_without_team_data(reports):
    return apply_function_data(lambda data: data[data['Last_name'] != 'Team'], reports)

# Вычислить эффективность
def get_eff(row, data):
    new_data = copy(data)
    
    new_data = new_data.fillna(0)
    new_data['Pos'] = list(map(lambda marks: marks[0] + marks[1], zip(new_data['#'], new_data['+'])))
    pos = new_data['Pos'].iloc[row]
    tot = new_data['Tot'].iloc[row]
    max_pos = max(new_data['Pos'])

    return Decimal((pos*pos)/(tot*max_pos) if max_pos > 0 else pos/tot).quantize(Decimal('1.00'))

# Получить вектор по отчёту для сета
def get_vector(data, set_value, player_cnt):
    element = filter_by_values(data, {'Set': [set_value], 'Skill': ['Reception']})

    if not element.empty:
        keys = ['Part']
        values = [element['Part'].iloc[0]]

        for i in range(player_cnt):
            if i < element.shape[0]:
                keys.extend([f'Number_{i + 1}', f'Ind_R_{i + 1}', f'Eff_R_{i + 1}'])
                values.extend([element['Number'].iloc[i], element['Ind.'].iloc[i], get_eff(i, element)])
            else:
                keys.extend([f'Number_{i + 1}', f'Ind_R_{i + 1}', f'Eff_R_{i + 1}'])
                values.extend([-1, 0, 0])
        
        vector = pd.DataFrame([values], columns=keys)
    else:
        vector = pd.DataFrame()

    return vector

# Получить выборку
def get_sample(reports, player_cnt):
    get_keys = ['Part', 'Number', 'Last_name', 'Skill', 'Set', 'Ind.', 'Tot', '+', '#']
    fill_keys = ['Skill', 'Player']
    new_reports = get_without_team_data(get_by_keys(get_keys, get_split_players(get_fill_general_nan(fill_keys, reports))))

    vectors = []
    for set_value in range(5):
        vectors.extend(apply_function_data(lambda data: get_vector(data, set_value + 1, player_cnt), new_reports))
    
    return pd.concat(filter(lambda data: not data.empty, vectors), ignore_index=True)

# Получить целевой вектор
def get_target_vector(data, player_cnt):
    players = list(map(lambda a: get_player_features(data, a + 1), range(player_cnt)))

    players_position = [player_cnt] + list(range(player_cnt))
    
    return reduce(lambda re_position, position: coach.get_replace(position, re_position, players), players_position) + 1

# Получить признаки игрока в выборке по позиции
def get_player_features(data, position):
    return {
        'part': data['Part'], 
        'num': data[f'Number_{position}'], 
        'ind': data[f'Ind_R_{position}'], 
        'eff': data[f'Eff_R_{position}']
    }

# Получить набор целевых векторов
def get_target(data, player_cnt):
    new_data = copy(data)
    new_data['Replace'] = list(map(lambda row: get_target_vector(data.iloc[row], player_cnt), range(data.shape[0])))

    return new_data

# Записать выборку в файл
def write_sample(data, path):
    writer = pd.ExcelWriter(path, engine='openpyxl')
    data.to_excel(writer, sheet_name='Sample', index=False)
    writer.save()

# Прочить выборку из файла
def read_sample(path):
    return pd.read_excel(path)

# Подсчитать кол-во меток классов
def get_class_cnt(target):
    classes = range(1, target.max() + 1)
    df = pd.DataFrame()
    
    for label in classes:
        df[label] = [target[target == label].shape[0]]

    return df
