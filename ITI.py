import numpy as np
import matplotlib.pyplot as plt
import random

def draw_iti_exponential(N, rate, low, high):

    samples = []

    while len(samples) < N:
        s = np.random.exponential(scale=1/rate, size=N) + low
        s = s[s <= high]
        samples.extend(s.tolist())

    intervals = np.array(samples[:N])

    return intervals

config = {
    'exp_name': 'AuditPrePro',
    'ISI': 0.65,
    'duration': 0.1,
    'feedback_duration': 2,
    'response_window': 0.65,
    'pre_cue': 0.75,
    'n_tones': 8,

}

duration = (config['n_tones']*config['duration']) + ((config['n_tones']-1)*config['ISI']) + config['pre_cue'] #+ config['response_window']
n_trials = 60
n_null = 15
nulls = [duration]*n_null
baseline = 15/60
initial_countdown = 4/60 

while True:
    iti = draw_iti_exponential((n_trials + n_null),0.9,2.25,4)
    sum_iti = sum(iti)/60
    block_duration = (duration*60)/60
    duration_null = (n_null*(duration))/60
    duration_feedback = (9*(2 + 2.5))/60
    print(f"mean ITI: {np.mean(iti)}")
    print(f"blck durations incl. baseline and countdown: {sum_iti + block_duration + duration_null + duration_feedback + baseline + initial_countdown}")
    
    if (sum_iti + block_duration + duration_null + duration_feedback + baseline + initial_countdown) <= 12 and np.mean(iti) > 2.7:
        break

plt.hist(iti, bins=75)
plt.title(f"block duration: {sum_iti + block_duration + duration_null + duration_feedback + baseline + initial_countdown}")
plt.ylabel('count')
plt.xlabel('ITI duration (s)')
plt.legend()
plt.savefig('ITI.png')

np.savetxt("itis.txt", iti)
np.savetxt("null_events.txt", nulls)