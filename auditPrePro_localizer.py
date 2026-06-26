# ----------------------------------------------------------#
# CHECK PSYCHOPY VERSION
# ----------------------------------------------------------#
import psychopy
print(f"Running PsychoPy {psychopy.__version__}")

# ----------------------------------------------------------#
# SET PSYCHOPY PREFS
# ----------------------------------------------------------#
from psychopy import prefs
prefs.hardware['keyboard'] = ['ptb']
prefs.hardware['audioLib'] = ['ptb']
prefs.hardware['audioLatencyMode'] = '4' # 4 is most strict
# prefs.hardware['audioDevice'] = 'Mac mini-Lautsprecher' # select speakers if necessary otherwise default

# ----------------------------------------------------------#
# IMPORTS
# ----------------------------------------------------------#
from psychopy import core
from psychopy import visual
from psychopy.hardware import keyboard
from psychopy import gui
from psychopy import monitors

import psychtoolbox as ptb
from psychtoolbox import PsychPortAudio

PsychPortAudio('Close')

import re
from collections import defaultdict

import pandas as pd
import numpy as np
import random

from scipy.io import wavfile
import scipy.signal as sps

from datetime import date, datetime
import os
import glob
import sys

# ----------------------------------------------------------#
# FUNCTIONS
# ----------------------------------------------------------#
def resample_sound(sound, current_fs, target_fs):
    '''resample sound from current to target sample rate

    Args:
        - sound: array containing sound
        - current_fs: current sample rate in Hz
        - target_fs: target sample rate in HZ

    Returns:
        - sound_resample: array containing resampled sound
    '''

    num_samples = int(len(sound) * target_fs / current_fs)
    sound_resample = sps.resample(sound, num_samples, axis=0)
    
    return sound_resample

def freq_key(s):
    return s[s.index("freq_"):]

def has_common_string(sublist1, sublist2):
    keys1 = {freq_key(s) for s in sublist1}
    keys2 = {freq_key(s) for s in sublist2}
    return bool(keys1 & keys2)

def shuffle_blocks(blocks):
    pairs = [blocks[i:i+2] for i in range(0, len(blocks), 2)]

    a = [pair[0] for pair in pairs]
    b = [pair[1] for pair in pairs]

    random.shuffle(a)

    while True:
        random.shuffle(b)
        if not has_common_string(a[-1], b[0]):
            break

    shuffled_blocks = a + b
    return shuffled_blocks
     
# ----------------------------------------------------------#
# SETUP
# ----------------------------------------------------------#

#---- open GUI to enter participant information
expName = 'AuditPrePro'
expInfo = {'participant':'', 'session':''} 

dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,title=expName)

if dlg.OK == False:
    core.quit()
expInfo['date'] = date.today()

participant = expInfo['participant']
session = expInfo['session']

config = {
    'exp_name': 'AuditPrePro_localizer',
    'screen_size': [1920, 1080],
    'screen_units': 'pix',
    'mouse_visible': False,
    'log_dir': 'logfiles_localizer/',
    'stim_dir': f"localizer_stims/",
    'ITI': 0.25,
    'IBI': 10,
    'N_blocks': 12,
    'N_in_blocks': 24, # for one second stimuli and 30 seconds block length
    'sample_rate': 48000,
    'trigger_key': '5',
    'N_triggers_wait': 5
}

os.makedirs(config['log_dir'], exist_ok=True) 

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#---- open a screen and test if refresh rate can be measured
mon = monitors.Monitor('tempMonitor') 
mon.setSizePix(config['screen_size']) 

win = visual.Window(
    size=(config['screen_size'][0],config['screen_size'][1]),
    fullscr=True,
    screen=1,
    units=config['screen_units'],
    color= 0,
    monitor = mon
)

refresh_rate = win.getActualFrameRate(
    nIdentical=20, nMaxFrames=200, nWarmUpFrames=10, threshold=1
)

if refresh_rate:
    print(f"Refresh rate: {refresh_rate:.2f} Hz")
else:
    print("Could not measure refresh rate.")

win.mouseVisible = config['mouse_visible']

escape_kb = keyboard.Keyboard(backend='ptb') 
task_kb = keyboard.Keyboard(backend='ptb') 
trigger_kb = keyboard.Keyboard(backend = 'ptb')
end_kb = keyboard.Keyboard(backend = 'ptb') 

# ----------------------------------------------------------#
# LOAD STIMULI
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Lade ...', wrapWidth=1500)
message.height = 60
message.draw()
win.flip()

#---- read in sound files and create an auditory buffer
wav_stimuli = [
    f for f in glob.glob(f"{config['stim_dir']}/*.wav")
    if "downsampled_noise" not in f
] # this should be list of all stimuli (w/o noise)

grouped_stimuli = defaultdict(list) 

# this should sort them by frequency (into blocks)
for file in wav_stimuli:
    filename = os.path.basename(file)

    match = re.search(r'\d+\.\d+', filename)

    if match:
        value = float(match.group())
        grouped_stimuli[value].append(file)

wav_stimuli = [grouped_stimuli[k] for k in sorted(grouped_stimuli)]

wav_stimuli_use = [random.sample(sublist, config['N_in_blocks']*2) for sublist in wav_stimuli]

wav_stimuli_use_blocks = []
for sublist in wav_stimuli_use:
    mid = len(sublist) // 2
    wav_stimuli_use_blocks.append(sublist[:mid])
    wav_stimuli_use_blocks.append(sublist[mid:])

#print(wav_stimuli_use_blocks)

wav_stimuli_order = shuffle_blocks(wav_stimuli_use_blocks)

min_gap = 4 # minimum gap between two noise corrupted sounds

for sublist in wav_stimuli_order:
    n = len(sublist)

    candidates = list(range(1, n - 1))

    selected_indices = []

    while candidates and len(selected_indices) < (config['N_in_blocks']/10): # percentage of noise corrputed trials
        
        idx = random.choice(candidates)
        selected_indices.append(idx)

        candidates = [
            i for i in candidates
            if abs(i - idx) > min_gap
        ]

    selected = set(selected_indices)

    for i, value in enumerate(sublist):
        if i in selected:
            if value.endswith(".wav"):
                sublist[i] = value.removesuffix(".wav") + "_noise.wav"

buffer_handles = []
pahandle = PsychPortAudio('Open', 1, 1, 4, config['sample_rate'], 1)

cf_sounds = [] 
sound_list = []
blocky =  []

#---- prepare prepape buffer handles
for block in range(0, config['N_blocks']):

    block_now = wav_stimuli_order[block]

    for trial in range(0, len(block_now)):

        waveform =[]
    
        samplerate, sound = wavfile.read(f'{block_now[trial]}')

        if samplerate != config['sample_rate']:
            sound = resample_sound(sound, samplerate, config['sample_rate'])

        sound = sound.astype(np.float32)
        buffer_handle = PsychPortAudio('CreateBuffer', [], sound)
        buffer_handles.append(buffer_handle)
        sound_list.append(sound)
        cf_sounds.append(block_now[trial])
        blocky.append(block)

        if trial != (len(block_now)-1):
            t = np.linspace(0, config['ITI'], int(config['sample_rate'] * config['ITI']), endpoint=False)
            itiy = np.zeros_like(t)
            itiy = itiy.astype(np.float32)
            buffer_handle = PsychPortAudio('CreateBuffer', [], itiy)
            buffer_handles.append(buffer_handle)
            sound_list.append(itiy)
            cf_sounds.append('iti')
            blocky.append(block)

    if block != config['N_blocks']:
        t = np.linspace(0, int(config['IBI']), int(config['sample_rate'] * int(config['IBI'])), endpoint=False)
        ibiy = np.zeros_like(t)
        ibiy = ibiy.astype(np.float32)
        sound_list.append(ibiy)
        cf_sounds.append('ibi')
        blocky.append(block)
    
    buffer_handle = PsychPortAudio('CreateBuffer', [], ibiy)
    buffer_handles.append(buffer_handle)

# ----------------------------------------------------------#
# INITIALIZE LOG VARIABLES
# ----------------------------------------------------------#
onsets_sounds = []
offsets_sounds = []
catchy = []
keyp = []
keyt = []
keyacc = []
feeddur = []
rt = []

# ----------------------------------------------------------#
# WAIT FOR SCANNER
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Warten auf den Scanner ...', wrapWidth=1500)
message.height = 60
message.draw()
win.flip()

trigger_key = [config['trigger_key']]
trigger_times = []

# ----------------------------------------------------------#
# START EXPERIMENT
# ----------------------------------------------------------#
for t in range(config['N_triggers_wait']): 
    trigger_keys = trigger_kb.waitKeys(keyList=trigger_key, waitRelease=False)
    trigger_time = ptb.GetSecs()
    trigger_times.append(trigger_time)

trigger_times_zero = [x - trigger_times[0] for x in trigger_times] 
first_trigger = trigger_times[0]    

timer_start = core.CountdownTimer(4.0)

while True:
    time_left = timer_start.getTime()

    if time_left <= 0.:
        break
    else:
        secs = int(time_left % 60)
        if secs != 0:
            message.text = f'{secs:01}'
            message.height = 100
            message.pos = (0, 0)
            message.draw()
            win.flip()
        if secs == 0:    
            message.text = '+'
            message.height = 100
            message.pos = (0, 0)
            message.draw()
            win.flip()


n_response_sounds = 2

pending_catch = None

feedback_onset = None
feedback_duration = 1

for i in range(0, len(buffer_handles)): 

    key_pressed = False
    key_rt_trial=None
    key_time = None
    accy = None

    PsychPortAudio('UseSchedule', pahandle, 1)  # 1 = replace current audio schedule
    PsychPortAudio('AddToSchedule', pahandle, buffer_handles[i]) # schedule
    
    onset = PsychPortAudio('Start', pahandle, 1, 0, 1) # measure onset trial based on PsychPortAudio
    onsets_sounds.append(onset-first_trigger)

    is_catch = "downsampled_noise" in cf_sounds[i]

    if is_catch:
        pending_catch = {
            "onset": onset,
            "expires": i + n_response_sounds,
            "responded": False
        }

    catchy.append(int(is_catch))

    while True:

        if 'escape' in [k.name for k in escape_kb.getKeys(['escape'], waitRelease=False)]:
            core.quit()
            sys.exit()

        response_keys = task_kb.getKeys(['9','1','2','3','4'], waitRelease=False)

        active_catch = (
            pending_catch is not None
            and i <= pending_catch["expires"]
            and not pending_catch["responded"]
        )

        # ----------------------------------------------
        # Valid hit to a catch trial
        # ----------------------------------------------
        if active_catch and response_keys and not key_pressed:

            key_pressed = True
            pending_catch["responded"] = True

            key_name = response_keys[0].name
            key_time = ptb.GetSecs()
            key_rt_trial = key_time - pending_catch["onset"]

            message.color = (0, 1, 0)
            accy = 1
            feedback_onset = key_time

        # ----------------------------------------------
        # False alarm (no active catch window)
        # ----------------------------------------------
        elif (not active_catch) and response_keys and not key_pressed:

            key_pressed = True

            key_name = response_keys[0].name
            key_time = ptb.GetSecs()
            key_rt_trial = key_time - onset

            message.color = (1, 0, 0)
            accy = 0
            feedback_onset = key_time

        # ----------------------------------------------
        # Feedback timing (always evaluated)
        # ----------------------------------------------
        if feedback_onset is not None:
            if ptb.GetSecs() - feedback_onset > feedback_duration:
                message.color = (1, 1, 1)
                feedback_onset = None


        # ----------------------------------------------
        # End of current sound
        # ----------------------------------------------
        if ptb.GetSecs() - onset > (len(sound_list[i]) / config['sample_rate']):

            if pending_catch is not None and i >= pending_catch["expires"]:
                pending_catch = None

            break

        message.draw()
        win.flip()

    # --------------------------------------------------
    # Save response
    # --------------------------------------------------
    rt.append(key_rt_trial)

    if key_rt_trial is not None:
        keyp.append(1)
        keyacc.append(accy)
        feeddur.append(feedback_duration)
        keyt.append(key_time - first_trigger)
    else:
        keyp.append(0)
        keyt.append(None)
        feeddur.append(None)
        keyacc.append(None)

    offset = PsychPortAudio('Stop', pahandle, 1)
    offsets_sounds.append(offset[-1] - first_trigger)

# ----------------------------------------------------------#
# COLLECT DATA IN BIDS FORMAT
# ----------------------------------------------------------#

data = pd.DataFrame({'onset': onsets_sounds,
                     'duration': [x - y for x, y in zip(offsets_sounds, onsets_sounds)],
                     'trial_type': catchy,
                     'response_time': rt,
                     'stim_file': cf_sounds,
                     'key_press': keyp,
                     'key_acc': keyacc,
                     'key_time': keyt,
                     'feedback_dur': feeddur,
                     'block': blocky,            
                     'trigger_times': [trigger_times] * int(len(onsets_sounds)),
                     'trigger_times_zero': [trigger_times_zero] * int(len(onsets_sounds)),
                     })

data.to_csv(f"{config['log_dir']}/sub-{participant}_ses-{session}_task-AuditPreProLocalizer_events-{date}.tsv", sep='\t', index=False)

# ----------------------------------------------------------#
# END EXPERIMENT
# ----------------------------------------------------------#
message.text = "Ende! Vielen Dank!"
message.draw()
win.flip()

end = ptb.GetSecs()
full_duration = (end-first_trigger)/60

end_keys = end_kb.waitKeys(keyList=['space'], waitRelease=False)

print(f"full duration experiment: {full_duration} minutes")

win.close()
core.quit()