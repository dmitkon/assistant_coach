def get_replace(position, re_position, players):
    if re_position == len(players):
        min_position = position
    else:
        min_position = re_position

    result = position if players[position].get('num') > -1 and \
                            (is_lower_eff(players[position].get('eff')) or
                            is_lower_eff_ind(players[position].get('eff'), players[position].get('ind')) or
                            is_lower_ind(players[position].get('ind'))) and \
                            is_no_individual(**players[position]) and \
                            (is_worst(players[position].get('eff'), 
                                        players[position].get('ind'), 
                                        players[min_position].get('eff'), 
                                        players[min_position].get('ind')) or re_position == len(players)) else re_position

    return result

def is_lower_eff(eff):
    return eff < 0.3

def is_lower_eff_ind(eff, ind):
    return eff < 0.5 and ind < 5.0

def is_lower_ind(ind):
    return ind < 4.0

def is_no_individual(part, num, eff, ind):
    return not (num == 13.0 and part < 3.0 or num == 18.0 and part < 2.0)

def is_worst(eff, ind, min_eff, min_ind):
    result = False
    if eff < min_eff:
        result = True
    elif eff == min_eff:
        if ind < min_ind:
            result = True
    return result
