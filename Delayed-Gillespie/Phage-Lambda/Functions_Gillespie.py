#!/usr/bin/env python
import Classes_Gillespie as Classy
import numpy as np
import heapq as q
import numba as nb
import time as t
import pickle as pkl
import os

'''
The current version of this code requires editing of this file in order to simulate different systems.
Lines throughout this file with the ~%~ symbol denote lines that must be altered.

TODO: Make queue recording a togglable feature
TODO: Reset time and track multiples of a modulus
'''

_PROP_TAB = None

def set_propensity_table(tab):
    """Set once in the parent so forked children share memory (COW)."""
    global _PROP_TAB
    _PROP_TAB = tab


def gillespie(reactions_list, stop_time, initial_state_vector, initial_queue=[], system_size=1):
    service_queue = initial_queue
    [state_vector, current_time, time_series] = initialize(initial_state_vector, reactions_list, system_size) # ~%~
    time_modulus = 0
    reset_time = 2**10
    indeces_of_resets = []
    while current_time + reset_time * time_modulus < stop_time:
        if current_time > reset_time:
            time_modulus += 1
            current_time = current_time - reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
            indeces_of_resets.append(len(time_series))
        cumulative_propensities = calculate_propensities(state_vector, reactions_list)
        next_event_time = draw_next_event_time(current_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            next_reaction = q.heappop(service_queue)
            state_vector += next_reaction[1]
            current_time = next_reaction[0]
            time_series.append({"time" : current_time, "state" : state_vector/system_size})
            continue
        current_time = next_event_time
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            time_series.append({"time" : current_time, "state" : state_vector/system_size}) # ~%~
        else:
            q.heappush(service_queue, (current_time + processing_time, next_reaction.change_vec))
    for index in indeces_of_resets:
        time_series[index:] = [{"time" : time_stamp["time"] + reset_time, "state" : time_stamp["state"]} for time_stamp in time_series[index:]]
    return time_series, service_queue


def gillespie_thresholding(reactions_list, stop_number, initial_state_vector, initial_queue=[], system_size=1, threshold=5, high_state=np.array([0.0735964308461,27.5042014971]), low_state=np.array([27.5042014971,0.0735964308461]), projection=np.array([[1,0],[0,1],[2,0],[0,2],[0,0],[2,0],[0,2]]), power_of_two=6):
    service_queue = initial_queue
    print("initializing...")
    real_start_time = t.time()
    [state_vector, current_time, current_state, propensity_table] = initialize_thresholding(initial_state_vector, reactions_list, system_size, high_state, threshold, projection, power_of_two) # ~%~
    bound = 2**power_of_two
    reset_time = 2**10
    reset = -2**14
    working_time = current_time
    hits = 0
    transitions = 0
    transitioned = False
    print("{} seconds taken to initialize".format(t.time()-real_start_time))
    print("burning in...")
    real_start_time = t.time()
    while reset < 0: ## Burn in
        if working_time > reset_time:
            reset += reset_time
            working_time -= reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
        if np.max(state_vector) < bound:
            cumulative_propensities = propensity_table[np.dot(np.array([0,1,2]),state_vector[-3:])][state_vector[0]][state_vector[1]][state_vector[2]][state_vector[3]] # ~%~
        else:
            cumulative_propensities = calculate_propensities(state_vector, reactions_list, system_size)
        next_event_time = draw_next_event_time(working_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            now_reaction = q.heappop(service_queue)
            working_time = now_reaction[0]
            current_time = working_time + reset
            state_vector += now_reaction[1]
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state, projection)
            if encoding[0]:
                current_state[0] = not current_state[0]
                current_state[1] = encoding[1]
            continue
        working_time = next_event_time
        current_time = working_time + reset
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state, projection)
            if encoding[0]:
                current_state[0] = not current_state[0]
                current_state[1] = encoding[1]
        else:
            q.heappush(service_queue, (working_time + processing_time, next_reaction.change_vec))
    time_series = [{"time" : working_time + reset, "state" : state_vector/system_size, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]}]
    print("took {} seconds to ".format(t.time()-real_start_time)+"burn in for {}".format(working_time+reset+2*reset_time)+" time units\n equilibriated state at start: {}\n".format(state_vector), end='\n')
    real_start_time = t.time()
    while transitions < stop_number:
        if working_time > reset_time:
            reset += reset_time
            working_time -= reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
        if np.max(state_vector) < bound:
            cumulative_propensities = propensity_table[np.dot(np.array([0,1,2]),state_vector[-3:])][state_vector[0]][state_vector[1]][state_vector[2]][state_vector[3]] # ~%~
        else:
            cumulative_propensities = calculate_propensities(state_vector, reactions_list, system_size)
        next_event_time = draw_next_event_time(working_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            now_reaction = q.heappop(service_queue)
            working_time = now_reaction[0]
            current_time = working_time + reset
            state_vector += now_reaction[1]
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state, projection)
            if encoding[0]:
                current_state[0] = not current_state[0]
                if current_state[0]:
                    hits += 1
                    transitioned = False
                elif current_state[1] == encoding[1]:
                    transitioned = False
                else:
                    transitioned = True
                    transitions += 1
                current_state[1] = encoding[1]
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]})
                if transitioned:
                    hits = 0
##            else:
##                if transitioned:
##                    transitioned = False
##                time_series.append({"time" : current_time, "state" : state_vector/system_size, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]}) #####
            continue
        working_time = next_event_time
        current_time = working_time + reset
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state, projection)
            if encoding[0]:
                current_state[0] = not current_state[0]
                if current_state[0]:
                    hits += 1
                    transitioned = False
                elif current_state[1] == encoding[1]:
                    transitioned = False
                else:
                    transitioned = True
                    transitions += 1
                current_state[1] = encoding[1]
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]})
                if transitioned:
                    hits = 0
##            else:
##                if transitioned:
##                    transitioned = False
##                time_series.append({"time" : current_time, "state" : state_vector/system_size, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]}) #####
        else:
            q.heappush(service_queue, (working_time + processing_time, next_reaction.change_vec))
    print("collected data for {} seconds".format(t.time()-real_start_time))
    return time_series


def gillespie_switching(reactions_list, stop_time, initial_state_vector, markov_rate=1, initial_markov_state=0, transition_matrix=np.array([[0,1],[1,0]], dtype=np.float32), initial_queue=[], system_size=1, transition_threshold=1, high_state=np.array([36]), low_state=np.array([7.2])):
    service_queue = initial_queue
    [state_vector, current_time, current_state, time_series, propensity_table, markov_state, markov_prev_state, markov_time, useful_matrix] = initialize_switching(initial_state_vector, transition_threshold, high_state, reactions_list, system_size, initial_markov_state, transition_matrix) # ~%~
    reset_time = 2**10
    reset = 0
    working_time = current_time
    while current_time < stop_time:
        if working_time > reset_time:
            reset += reset_time
            working_time -= reset_time
            markov_time -= reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
        cumulative_propensities = propensity_table[state_vector[0]] # ~%~
        next_event_time = draw_next_event_time(working_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            now_reaction = q.heappop(service_queue)
            working_time = now_reaction[0]
            current_time = working_time + reset
            if working_time >= markov_time:
                [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
                for update in markov_history:
                    time_series.append({"time" : update["time"]+reset, "state" : state_vector/system_size, "markov state" : update["state"], "transition" : False})
            state_vector += now_reaction[1]
            if transitioned(state_vector, current_state, system_size, transition_threshold, low_state, high_state):
                current_state = not current_state
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "transition" : True})
            else:
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "transition" : False})
            continue
        working_time = next_event_time
        current_time = working_time + reset
        if working_time >= markov_time:
            [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
            for update in markov_history:
                time_series.append({"time" : update["time"]+reset, "state" : state_vector/system_size, "markov state" : update["state"], "transition" : False})
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution(markov_prev_state)
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            if transitioned(state_vector, current_state, system_size, transition_threshold, low_state, high_state):
                current_state = not current_state
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "transition" : True})
            else:
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "transition" : False}) # ~%~
        else:
            q.heappush(service_queue, (working_time + processing_time, next_reaction.change_vec))    
    return time_series, service_queue


def gillespie_switching_transitions(reactions_list, stop_number, initial_state_vector, markov_rate=1, initial_markov_state=np.random.randint(2), transition_matrix=np.array([[0,1],[1,0]], dtype=np.float32), initial_queue=[], system_size=1, transition_threshold=1, high_state=np.array([36]), low_state=np.array([7.2])):
    service_queue = initial_queue
    [state_vector, current_time, current_state, time_series, propensity_table, markov_state, markov_prev_state, markov_time, useful_matrix] = initialize_switching(initial_state_vector, transition_threshold, high_state, reactions_list, system_size, initial_markov_state, transition_matrix) # ~%~
    reset_time = 2**10
    reset = 0
    working_time = current_time
    transitions = 0
    while transitions < stop_number:
        if working_time > reset_time:
            reset += reset_time
            working_time -= reset_time
            markov_time -= reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
        cumulative_propensities = propensity_table[state_vector[0]][state_vector[1]] # ~%~
        next_event_time = draw_next_event_time(working_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            now_reaction = q.heappop(service_queue)
            working_time = now_reaction[0]
            current_time = working_time + reset
            if working_time >= markov_time:
                [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
            state_vector += now_reaction[1]
            if transitioned(state_vector, current_state, system_size, transition_threshold, low_state, high_state):
                current_state = not current_state
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "high_low" : current_state})
                transitions += 1
            continue
        working_time = next_event_time
        current_time = working_time + reset
        if working_time >= markov_time:
            [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution(markov_prev_state)
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            if transitioned(state_vector, current_state, system_size, transition_threshold, low_state, high_state):
                current_state = not current_state
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "high_low" : current_state})
                transitions += 1
        else:
            q.heappush(service_queue, (working_time + processing_time, next_reaction.change_vec))    
    return time_series


def gillespie_switching_thresholding(reactions_list, stop_number, initial_state_vector, markov_rate=1, initial_markov_state=np.random.randint(2), transition_matrix=np.array([[0,1],[1,0]], dtype=np.float32), initial_queue=[], system_size=1, threshold=6, high_state=np.array([1.53493,30.10547]), low_state=np.array([30.10547,1.53493])):
    service_queue = initial_queue
    [state_vector, current_time, current_state, propensity_table, markov_state, markov_prev_state, markov_time, useful_matrix] = initialize_thresholding(initial_state_vector, reactions_list, system_size, initial_markov_state, transition_matrix, high_state, threshold) # ~%~
    reset_time = 2**10
    reset = -2**11
    working_time = current_time
    hits = 0
    transitions = 0
    transitioned = False
    real_start_time = t.time()
    while reset < 0: ## Burn in
        if working_time > reset_time:
            reset += reset_time
            working_time -= reset_time
            markov_time -= reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
        cumulative_propensities = propensity_table[state_vector[0]][state_vector[1]] # ~%~
        next_event_time = draw_next_event_time(working_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            now_reaction = q.heappop(service_queue)
            working_time = now_reaction[0]
            current_time = working_time + reset
            if working_time >= markov_time:
                [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
            state_vector += now_reaction[1]
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state)
            if encoding[0]:
                current_state[0] = not current_state[0]
                current_state[1] = encoding[1]
            continue
        working_time = next_event_time
        current_time = working_time + reset
        if working_time >= markov_time:
            [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution(markov_prev_state)
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state)
            if encoding[0]:
                current_state[0] = not current_state[0]
                current_state[1] = encoding[1]
        else:
            q.heappush(service_queue, (working_time + processing_time, next_reaction.change_vec))
    time_series = [{"time" : working_time + reset, "state" : state_vector/system_size, "markov state" : markov_prev_state, "intermediate" : current_state}]
    print("burned in for {}".format(working_time+reset+2*reset_time)+" time units\n equilibriated state at start: {}\n".format(state_vector), end='\n')
    while transitions < stop_number:
        if working_time > reset_time:
            reset += reset_time
            working_time -= reset_time
            markov_time -= reset_time
            service_queue = [(reaction[0]-reset_time,reaction[1]) for reaction in service_queue]
        cumulative_propensities = propensity_table[state_vector[0]][state_vector[1]] # ~%~
        next_event_time = draw_next_event_time(working_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            now_reaction = q.heappop(service_queue)
            working_time = now_reaction[0]
            current_time = working_time + reset
            if working_time >= markov_time:
                [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
            state_vector += now_reaction[1]
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state)
            if encoding[0]:
                current_state[0] = not current_state[0]
                if current_state[0]:
                    hits += 1
                    transitioned = False
                elif current_state[1] == encoding[1]:
                    transitioned = False
                else:
                    transitioned = True
                    transitions += 1
                current_state[1] = encoding[1]
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]})
                if transitioned:
                    hits = 0
            continue
        working_time = next_event_time
        current_time = working_time + reset
        if working_time >= markov_time:
            [markov_time, markov_prev_state, markov_state, markov_history] = update_markov(markov_prev_state, markov_state, markov_time, working_time, markov_rate, useful_matrix)
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution(markov_prev_state)
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            encoding = hit_unhit(state_vector, current_state, system_size, threshold, low_state, high_state)
            if encoding[0]:
                current_state[0] = not current_state[0]
                if current_state[0]:
                    hits += 1
                    transitioned = False
                elif current_state[1] == encoding[1]:
                    transitioned = False
                else:
                    transitioned = True
                    transitions += 1
                current_state[1] = encoding[1]
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "markov state" : markov_prev_state, "transition" : transitioned, "hits" : hits, "intermediate" : current_state[0]})
                if transitioned:
                    hits = 0
        else:
            q.heappush(service_queue, (working_time + processing_time, next_reaction.change_vec))
    print(t.time()-real_start_time)
    return time_series


def initialize(initial_state_vector, reactions_list, system_size):
    state_vector = initial_state_vector
    current_time = 0
    time_series = [{"time" : 0, "state" : state_vector, "transition" : False}] # ~%~
##    propensity_table = [[] for num0 in range(128)] # ~%~
##    for num0 in range(128): # ~%~
##        for num1 in range(128):
##            propensity_table[num0].append(calculate_propensities(np.array([num0,num1]), reactions_list, system_size)) # ~%~
    return [state_vector, current_time, time_series]


def initialize_switching(initial_state_vector, threshold, high_state, reactions_list, system_size, initial_markov_state, transition_matrix):
    state_vector = initial_state_vector
    current_state = (np.linalg.norm(state_vector/system_size - high_state) <= threshold)
    current_time = 0
    markov_state = initial_markov_state
    markov_prev_state = markov_state
    markov_time = 0
    useful_matrix = transition_matrix.copy()
    for index in range(np.shape(useful_matrix)[0]):
        useful_matrix[index] = np.cumsum(transition_matrix[index])
    propensity_table = [[] for num in range(4096)]
    for num0 in range(4096):
        for num1 in range(4096):
            propensity_table[num0].append(calculate_propensities(np.array([num0,num1]), reactions_list, system_size))
    time_series = [{"time" : 0, "state" : state_vector, "markov state" : markov_state, "transition" : False}] # ~%~
    return [state_vector, current_time, current_state, time_series, propensity_table, markov_state, markov_prev_state, markov_time, useful_matrix]


def initialize_thresholding(initial_state_vector, reactions_list, system_size,
                            high_state, threshold, projection, power_of_two=5):
    state_vector = initial_state_vector
    current_state = [False, (np.linalg.norm(np.matmul(state_vector/system_size,projection)
                                            - high_state) <= threshold)]
    current_time = 0

    global _PROP_TAB
    if _PROP_TAB is not None:
        propensity_table = _PROP_TAB
    else:
        if os.path.exists("./propensity_table_{}.pkl.gz".format(power_of_two)):
            os.system("gunzip ./propensity_table_{}.pkl.gz".format(power_of_two))
            with open("./propensity_table_{}.pkl".format(power_of_two), "rb") as props:
                propensity_table = pkl.load(props)
            os.system("gzip ./propensity_table_{}.pkl".format(power_of_two))
        else:
            size = 2**power_of_two
            propensity_table = [[[[[] for num2 in range(size)] for num1 in range(size)] for num0 in range(size)],
                                [[[[] for num2 in range(size)] for num1 in range(size)] for num0 in range(size)],
                                [[[[] for num2 in range(size)] for num1 in range(size)] for num0 in range(size)]]
            for num0 in range(size):
                for num1 in range(size):
                    for num2 in range(size):
                        for num3 in range(size):
                            propensity_table[0][num0][num1][num2].append(calculate_propensities(np.array([num0,num1,num2,num3,1,0,0]), reactions_list, system_size))
                            propensity_table[1][num0][num1][num2].append(calculate_propensities(np.array([num0,num1,num2,num3,0,1,0]), reactions_list, system_size))
                            propensity_table[2][num0][num1][num2].append(calculate_propensities(np.array([num0,num1,num2,num3,0,0,1]), reactions_list, system_size))
            with open("./propensity_table_{}.pkl".format(power_of_two), "wb") as props:
                pkl.dump(propensity_table, props)
            os.system("gzip ./propensity_table_{}.pkl".format(power_of_two))
    return [state_vector, current_time, current_state, propensity_table]


def initialize_transitions(initial_state_vector, threshold, high_state, reactions_list, system_size):
    state_vector = initial_state_vector
    current_state = (np.linalg.norm(state_vector/system_size-high_state)<=threshold)
    current_time = 0
    time_series = [{"time" : 0, "high_low" : current_state}] # ~%~
    propensity_table = [[] for num0 in range(128)] # ~%~
    for num0 in range(128): # ~%~
        for num1 in range(128):
            propensity_table[num0].append(calculate_propensities(np.array([num0,num1]), reactions_list, system_size)) # ~%~
    return [state_vector, current_state, current_time, time_series, propensity_table]


def calculate_propensities(x, reactions_list, system_size=1):
    propensities = np.zeros(np.shape(reactions_list), dtype=np.float32)
    for index in range(np.size(reactions_list)):
        propensities[index] = np.float32(reactions_list[index].propensity(x/system_size)*system_size)
    return np.cumsum(propensities)


def reaction_will_complete(queue, next_event_time):
    if queue:
        if next_event_time > queue[0][0]:
            return True
    return False

'''
The transitioned function must be edited for different systems.
'''

def transitioned(state, side, size, thresh, low, high): # ~%~
    if side:
        return (np.linalg.norm(state/size - low) <= thresh)
    else:
        return (np.linalg.norm(state/size - high) <= thresh)


def hit_unhit(state, lhi, size, thresh, low, high, proj):
    shadow = np.matmul(state/size, proj)
    if lhi[0]:
        if np.linalg.norm(shadow-low) <= thresh:
            return [1, 0]
        elif np.linalg.norm(shadow-high) <= thresh:
            return [1, 1]
        else:
            return [0, lhi[1]]
    elif abs(shadow[0]-shadow[1])/size <= thresh:
        return [1, lhi[1]]
    else:
        return [0, lhi[1]]


@nb.jit("float64(float64, float32)")
def draw_next_event_time(current_time, summed_propensities):
    return current_time + np.random.exponential(scale=(1 / summed_propensities))


@nb.jit("int16(float32[:])")
def choose_reaction(cumulative_propensities):
    u = np.random.uniform()
    next_reaction_index = min(
        np.where(cumulative_propensities >= cumulative_propensities[-1] * u)[0])
    return next_reaction_index


@nb.jit("int16(float32[:])")
def next_state(cumulative_probability):
    u = np.random.uniform()
    next_index = min(np.where(cumulative_probability >= u)[0])
    return next_index


def update_markov(prev_state, state, time, target_time, rate, matrix):
    now = time
    ps = prev_state
    st = state
    events = []
    while now <= target_time:
        events.append({"time" : now, "state" : st})
        now = draw_next_event_time(now, rate)
        ps = st
        st = next_state(matrix[st])
    return [now, ps, st, events]


def recursive_list_formation(parameter_ranges, long_list):
    if len(parameter_ranges) != 0:
        par_range = len(parameter_ranges[0])
        current_length = len(long_list)
        long_list = long_list * par_range
        for index1 in range(par_range):
            for index2 in range(index1 * current_length, (index1 + 1) * current_length):
                long_list[index2] = long_list[index2] + [parameter_ranges[0][index1]]
        return recursive_list_formation(parameter_ranges[1:], long_list)
    else:
        return long_list


def list_for_parallelization(parameter_ranges):
    long_list = []
    for index in range(len(parameter_ranges[0])):
        long_list.append([parameter_ranges[0][index]])
    return recursive_list_formation(parameter_ranges[1:], long_list)

