
#---- set preferences first --> ATTENTION: manually use ptb backend for keyboard later on anyways on Mac!
from psychopy import prefs
#prefs.general['version'] = '2025.1.1'
prefs.hardware['keyboard'] = 'ptb'
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4' # could also use 4 but then no fallback in case of small deviation
#prefs.hardware['audioDevice'] = 'Kopfhörer' # cable headphones
#prefs.hardware['audioDevice'] = 'Mac mini-Lautsprecher' # mac mini speakers
#prefs.hardware['audioDevice'] = 'Speakers (Realtek HD Audio output)'
#prefs.hardware['audioDevice'] = 'Default' # cable headphones

#---- check psychopy version
#from psychopy import useVersion
#useVersion('2025.1.1')

#---- imports
import psychopy
print(f"Running PsychoPy {psychopy.__version__}")

import psychtoolbox as ptb
from psychtoolbox import audio
from psychtoolbox import PsychPortAudio
PsychPortAudio('Close')

import sounddevice as sd

from psychopy import core
from psychopy import visual
from psychopy.hardware import keyboard
from psychopy import gui
from psychopy import monitors

import pandas as pd
import numpy as np
from datetime import date
import os

import sys
sys.path.append('/Users/steinj/Documents/2025_paradigm/thorns')
#import thorns as th

exp_version = 'behavioral' # or 'scanner'

#---- create logfile directory if not already existing
os.makedirs('logfiles_behavioral/', exist_ok=True)

#---- open GUI to enter participant information
expName = 'AuditPrePro'
expInfo = {'participant':'', 'session':'', 'run': ''} # enter participant and session info as defined in generate_task_sequences.py

dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,title=expName)
if dlg.OK == False:
    core.quit()
expInfo['date'] = date.today()

participant = expInfo['participant']
session = expInfo['session']
run_info = expInfo['run']

date = date.today()
date = date.strftime("%Y-%m-%d")

trial_list_dir = 'trial_lists/'

#---- open a screen and test if refresh rate can be measured
mon = monitors.Monitor('tempMonitor')  # name can be anything
mon.setSizePix([3840, 2160])  

win = visual.Window(
    size=(3840, 2160),
    fullscr=True,
    screen=1,
    units='pix',
    color= 0,
    monitor = mon
)

win.mouseVisible = False

refresh_rate = win.getActualFrameRate(
    nIdentical=20, nMaxFrames=200, nWarmUpFrames=10, threshold=1
)

if refresh_rate:
    print(f"Refresh rate: {refresh_rate:.2f} Hz")
else:
    print("Could not measure refresh rate.")
    
message = visual.TextStim(win, text='Starte Experiment ...', wrapWidth=2000)
message.height = 100
message.draw()
win.flip()

circle = visual.Circle(win, radius=50, fillColor='white', lineColor='white')
triangle = visual.ShapeStim(win, vertices=[(-50, -50), (0, 50), (50, -50)], fillColor='white', lineColor='white')
square = visual.ShapeStim(win, vertices=[(-50, -50), (-50, 50), (50, 50), (50, -50)], fillColor='white', lineColor=None)

#---- define variables to record
onset_sound = []
onset_tones = []

duration_sound = [] # based on theoretical duration
duration_sound_getsecs = [] # based on getsecs

offset_trial = [] # based on theroretical duretion

offset_sound = [] # based on theoretical duration
offset_sound_getsecs = [] # based on getsecs

onset_iti_list = [] # based on theoretical duration
onset_iti_list_getsecs = [] # based on getsecs

offset_iti_list = [] # based on theoretical duration
offset_iti_list_getsecs = [] # based on getsecs

ITI_list = [] # theoretical ITIs

tau = [] # tau

frequency = [] # tone frequency in Hz
lim_std = [] # mu std
lim_dev = [] # mu dev

rule = [] # rule
cue = [] # cue
dpos = [] # deviant positions
tone_type = [] # std or deviant

runs = []
trial_nr = []

feedback_trialy = []
feedback_trial = False

#---- initialize all keyboards
escape_kb = keyboard.Keyboard(backend='ptb') # set keyboard that can be used to stop the experiment at any point
escape_kb.waitKeys(maxWait=1) 

trigger_kb = keyboard.Keyboard(backend = 'ptb') # trigger kb
trigger_kb.waitKeys(maxWait=1)  

response_kb = keyboard.Keyboard(backend = 'ptb') # response kb
response_kb.waitKeys(maxWait=1)  

#---- functions to generate soundwaves for the sounds and ISI/ITI
def generate_timbre_waveform(frequency = 1450.0, harmonics = [(1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25), (5, 0.2)], duration = 0.1, sample_rate= 48000, ramp_time = 0.01, hanning = True):

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    for multiple, amplitude in harmonics:
        waveform += amplitude * np.sin(2 * np.pi * frequency * multiple * t)

    waveform /= np.max(np.abs(waveform))
    
    if hanning == False:
        ramp_samples = int(sample_rate * ramp_time)
        ramp = np.linspace(0, 1, ramp_samples)
        envelope = np.ones_like(waveform)
        envelope[:ramp_samples] *= ramp
        envelope[-ramp_samples:] *= ramp[::-1]
    
    elif hanning == True: 
        ramp_samples = int(sample_rate * ramp_time)
        full_window = np.hanning(2 * ramp_samples)
        ramp_in = full_window[:ramp_samples]
        ramp_out = full_window[ramp_samples:]

        envelope = np.ones_like(waveform)
        envelope[:ramp_samples] *= ramp_in
        envelope[-ramp_samples:] *= ramp_out

    return waveform * envelope * 0.3
    

def generate_isi(duration = 0.65, sample_rate = 48000):
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform_isi = np.zeros_like(t)

    return waveform_isi

#---- define important variables

# possible implementation of different harmonics
harmonics_dict =  {
    "pure": [(1.0, 1.0)],
    "minor third": [(1, 1.0), (6/5, 0.5)], # 1/n
    "major third": [(1, 1.0), (5/4, 0.5)], # 1/n
    "perfect fourth": [(1, 1.0), (4/3, 0.5)], # 1/n
    "perfect fifth": [(1, 1.0), (3/2, 0.5)], # 1/n
    "minor sixth": [(1, 1.0), (8/5, 0.5)], # 1/n
    "major sixth": [(1, 1.0), (5/3, 0.5)], # 1/n
    "octave": [(1, 1.0), (2, 0.5)], # 1/n
    "1/n": [(n, 1/n) for n in range(1, 6)], # pure 1/n
    "1/n^2": [(n, 1/(n**2)) for n in range(1, 6)],
    "e^(-0.5(n-1))": [(n, np.exp(-0.5*(n - 1))) for n in range(1, 6)],
    "odd 1/n": [(n, 1/n) for n in range(1, 10, 2)]
}

harmonics = harmonics_dict['1/n'] # choose implementation of harmonics from harmonics dict

sample_rate = 48000 # TODO: adjust to final device used!
ramp_time = 0.01 # ramp time for on/off ramps
stim_dur = 0.1 # total stmulus duration incl. on/off ramps
isi_dur = 0.65 # inter-stimulus-interval

pre_cue = 0.75 # cueing time frame before onset of first tone in sequence

response_window = 0.75 # window to still respond after last deviant
trial_duration = (8*stim_dur) + (7*isi_dur) + pre_cue # trial duration (incl. pre-cue)
key_pos = ['v','y','u','i','l'] # response keys deviant position
feedback_duration = 2 # feedback duration

#---- read in the pre-created trial lists
trials = pd.read_csv(f'{trial_list_dir}sub-{participant}/sub-{participant}_ses-{session}_run-{run_info}_trials.csv')
trials['dpos'] = trials['dpos'].fillna(0).astype(int)
n_trials = len(trials)/8

rts_getsecs_trial = [None]*int(n_trials) # RT based on ptb.getSecs() from onset trial
rts_getsecs_dev = [None]*int(n_trials) # RT based on ptb.getSecs() from onset deviant
keys_pressed = [None]*int(n_trials) # pressed key
performance = [None]*int(n_trials) # correct?
confidence = [None]*int(n_trials) # confidence ratings

#---- more important variables
run_step = len(trials) # n trials per run --> attention: is just len(trials if runs are run separately)
run_indices = [(0, run_step)] # indices new runs

len_waveform = [] # len waveform used to compute duration of audio playback
buffer_handles = [] # buffer for audido playback

#---- open audio port
pahandle = PsychPortAudio('Open', [], 1, 4, sample_rate, 1) # TODO: watch out, might need to adjust to final machine!

#---- load experiment: create all trials in advance from previously generated trial_list
for i in range(0, len(trials), 8):
    
    message = visual.TextStim(win, text='Lade Stimuli ...', wrapWidth=2000)
    message.height = 100
    message.draw()
    win.flip()
    
    waveform = []
    oddball_trial = trials['frequency'][i:i + 8]
    
    dpos_trial = trials['dpos'][i:i+8]
    dposy = dpos_trial.iloc[0]
    dposy = int(dposy)
    dpos.append(dposy)
    
    run_trial = trials['run_n'][i:i+8]
    runy = run_trial.iloc[0]
    runs.append(runy)
    
    rule_trial = trials['rule'][i:i+8]
    ruly = rule_trial.iloc[0]
    rule.append(ruly)

    cue_trial = trials['cue'][i:i+8]
    cuey = cue_trial.iloc[0]
    cue.append(cuey)
    
    tones_trial = np.zeros(8)
    if dposy != 0:
        tones_trial[dposy] = 1
    tone_type.append(tones_trial.astype(int))
    
    trial_trial = trials['trial_n'][i:i+8]
    trialy = trial_trial.iloc[0]
    trial_nr.append(trialy)
    
    tau_std_trial = trials['tau_std'][i:i+8]
    tauy = tau_std_trial.iloc[0]
    tau.append(tauy)
    
    lim_std_trial = trials['lim_std'][i:i+8]
    lim_stdy = lim_std_trial.iloc[0]
    lim_std.append(lim_stdy)
    
    lim_dev_trial = trials['lim_dev'][i:i+8]
    lim_devy = lim_dev_trial.iloc[0]
    lim_dev.append(lim_devy)
            
    ITI = trials['ITI'][i]
    ITI_list.append(ITI)    
   
    tone_count = 0

    wave = generate_isi(pre_cue, sample_rate) # generate silence during which pre-cueing takes place
    waveform.append(wave)
    
    for s in oddball_trial:
        tone_count += 1
        frequency.append(round(s,2))
        wave = generate_timbre_waveform(round(s,2), harmonics, stim_dur, sample_rate, ramp_time)
        waveform.append(wave)
        
        if tone_count < 8:
            isi = generate_isi(isi_dur, sample_rate)
            waveform.append(isi)
        elif tone_count == 8:
            iti_wave = generate_isi(ITI, sample_rate)
            waveform.append(iti_wave)
        
    waveform = np.concatenate(waveform)
    len_waveform.append(len(waveform))
    
    waveform = waveform.astype(np.float32)
    buffer_handle = PsychPortAudio('CreateBuffer', [], waveform)
    buffer_handles.append(buffer_handle)

#---- wait for space press to start

if exp_version == 'scanner':
    message = visual.TextStim(win, text='Warte auf Scanner...', wrapWidth=2000)
    message.height = 100
    message.draw()
    win.flip()

    trigger_list = ['5'] #for 7T
    n_triggers_to_wait = 5 # how many kay presses to wait for
    trigger_times = [] # trigger_times[] would be the time to subtract from all onsets, save all 5 in case we need to get rid of non-steady-state images

    for t in range(n_triggers_to_wait): # start on the first space press for behavioral experiment
        trigger_keys = trigger_kb.waitKeys(keyList=trigger_list, waitRelease=False) # TODO: adjust for scanning
        trigger_time = ptb.GetSecs()
        trigger_times[t] = trigger_time

elif exp_version == 'behavioral':
    message = visual.TextStim(win, text='Drücke die Leertaste, um zu starten...', wrapWidth=2000)
    message.height = 100
    message.draw()
    win.flip()

    trigger_list = ['space'] #for 7T
    n_triggers_to_wait = 1 # how many kay presses to wait for

    for t in range(n_triggers_to_wait): # start on the first space press for behavioral experiment
        trigger_keys = trigger_kb.waitKeys(keyList=trigger_list, waitRelease=False) # TODO: adjust for scanning
        trigger_time = ptb.GetSecs()
        
#---- show fixation cross
message.text = '+'
message.height = 100
message.pos = (0, 0)
message.draw()
win.flip()

# for too fast responses
message_too_fast = visual.TextStim(win, text=' ', wrapWidth=2000)
message_too_fast.height = 50
message_too_fast.color = (1,0,0)
message_too_fast.pos = (0, 150)

#---- feedback for too slow
feed = visual.TextStim(win, text='zu langsam', wrapWidth=2500)
feed.color = (1, 0, 0)
feed.height = 50
feed.pos = (0, 150)

#---accumulates accuracy feedback
feed_acc = visual.TextStim(win, text=' ', wrapWidth=2500)
feed_acc.color = (1, 1, 1)
feed_acc.height = 100
feed_acc.pos = (0, 0)

#---- play oddball sequences and record logfiles separately for each run
for i in range(0, int(n_trials) + 1):

    # add indicator for feedback trials (every 6th trial but not at end of run)
    if (i+1) % 6 == 0 and i > 0 and i != n_trials:
        feedback_trial = True
        feedback_trialy.append(1)
    else:
        feedback_trial = False
        feedback_trialy.append(0)
      
    # if at the end, append last run number
    if i == n_trials:
        runs.append(5)
    
    # whenever run changes:
    if i > 0 and (runs[i] != runs[i-1] or i == n_trials):

        trial_run = run_indices[0]
        trial_run_slow = (np.array(trial_run)/8).astype(int)                

        # compute accuracy for full run to present to participant
        #accuracy_run = np.abs(((len(np.where(np.array(performance[trial_run_slow[0]:trial_run_slow[1]]) == 1)[0]) + len(np.where(np.array(performance[trial_run_slow[0]:trial_run_slow[1]]) == 4)[0]))/len(performance[trial_run_slow[0]:trial_run_slow[1]]))*100)
        #message.text = f"% korrekte Positionen in diesem Durchgang: {accuracy_run: .2f}\n\n\nKurze Pause!\n\n\nWeiter geht's mit der Leertaste"
        #message.draw()
        #win.flip()
        
        # create datafile for previous run
        data = []

        print(n_trials,trial_run_slow[0],trial_run_slow[1])
        
        # write datafile (TODO: make this more BIDS-like & remove unnecessary stuff!)
        data = pd.DataFrame({
            'onset': np.repeat(onset_sound[trial_run_slow[0]:trial_run_slow[1]],8), # onset sound sequence (based on PsychPortAudio)
            'onset_tone': np.concatenate(onset_tones)[trial_run[0]:trial_run[1]], # onset of each tone based on theoretical duration and isi
            'duration': np.repeat(duration_sound[trial_run_slow[0]:trial_run_slow[1]],8), # theooretical duration sound sequence 
            'duration_getsecs': np.repeat(duration_sound_getsecs[trial_run_slow[0]:trial_run_slow[1]],8), # duration based on getsecs()
            'offset_trial': np.repeat(offset_trial[trial_run_slow[0]:trial_run_slow[1]],8), # measured based on PsychPortAudio
            'offset_sound': np.repeat(offset_sound[trial_run_slow[0]:trial_run_slow[1]],8), # based on theoretical
            'offset_sound_getsecs': np.repeat(offset_sound_getsecs[trial_run_slow[0]:trial_run_slow[1]],8), # based on getsecs()
            'onset_iti': np.repeat(onset_iti_list[trial_run_slow[0]:trial_run_slow[1]],8), # based on theoretical duration
            'onset_iti_getsecs': np.repeat(onset_iti_list_getsecs[trial_run_slow[0]:trial_run_slow[1]],8), # based on getsecs()
            'offset_iti': np.repeat(offset_iti_list[trial_run_slow[0]:trial_run_slow[1]],8), # based on theoretical duration
            'offset_iti_getsecs': np.repeat(offset_iti_list_getsecs[trial_run_slow[0]:trial_run_slow[1]],8), # based on getsecs()
            'ITI': np.repeat(ITI_list,8)[trial_run[0]:trial_run[1]], # theoretical ITI
            'rt_getsecs_trial': np.repeat(rts_getsecs_trial[trial_run_slow[0]:trial_run_slow[1]],8), # response time based on getsecs from trial start
            'rt_getsecs_dev': np.repeat(rts_getsecs_dev[trial_run_slow[0]:trial_run_slow[1]],8), # response time based on getsecs from trial start
            'tau_std': np.repeat(tau,8)[trial_run[0]:trial_run[1]], # tau std
            'frequency': frequency[trial_run[0]:trial_run[1]], # sound frequency (base)
            'lim_std': np.repeat(lim_std,8)[trial_run[0]:trial_run[1]], # mu std
            'lim_dev': np.repeat(lim_dev,8)[trial_run[0]:trial_run[1]], # mu dev
            'key_pressed': np.repeat(keys_pressed[trial_run_slow[0]:trial_run_slow[1]],8), # pressed key
            'correct': np.repeat(performance[trial_run_slow[0]:trial_run_slow[1]],8), # key press correct?
            'confidence': np.repeat(confidence[trial_run_slow[0]:trial_run_slow[1]],8), # confidence
            'rule': np.repeat(rule,8)[trial_run[0]:trial_run[1]], # rule no
            'dpos': np.repeat(dpos,8)[trial_run[0]:trial_run[1]], # deviant position
            'cue': np.repeat(cue,8)[trial_run[0]:trial_run[1]],
            'trial_type': np.concatenate(tone_type)[trial_run[0]:trial_run[1]], # standard or deviant?
            'runs': np.repeat(runs,8)[trial_run[0]:trial_run[1]], # run no
            'trial_no': np.repeat(trial_nr,8)[trial_run[0]:trial_run[1]], # trial no
            'feedback_trial': np.repeat(feedback_trialy[trial_run_slow[0]:trial_run_slow[1]],8)
        })
        
        # save logfiles
        if runs[i] == 5:
             data.to_csv(f'logfiles_behavioral/sub-{participant}-ses-{session}-run-{run_info}-events_{date}.tsv', sep='\t', index=False)
             break
        else:
            data.to_csv(f'logfiles_behavioral/sub-{participant}-ses-{session}-run-{run_info}-events_{date}.tsv', sep='\t', index=False)
        
            # wait for space key press to initialize next run
            pause_kb = keyboard.Keyboard(backend = 'ptb')
            pause_keys = pause_kb.waitKeys(keyList=['space'], waitRelease=False)
            message.text = '+'
            message.draw()
            win.flip() 

    #---- prepare sound playback
    phase ='stimulus'
    key_pressed = False
    feedback_recorded = False
    feedback_start = None
    response_start = None
    key_name = None
    key_rt_trial = None
    key_rt_dev = None

    # play audio sequences from buffer
    PsychPortAudio('UseSchedule', pahandle, 1)  # 1 = replace current schedule
    PsychPortAudio('AddToSchedule', pahandle, buffer_handles[i])
    onset = PsychPortAudio('Start', pahandle, 1, 0, 1) # measure onset trial based on PsychPortAudio
    onset_soundstim = onset + pre_cue

    time_to_dev = (dpos[i]*stim_dur) + (dpos[i]*isi_dur)
    time_to_dev_all = [((np.unique(dpos)[j-2]*stim_dur) + (np.unique(dpos)[j-2]*isi_dur)) for j in np.unique(dpos)]
    time_to_dev_3 = (2*stim_dur) + (2*isi_dur)
    
    response_kb.clearEvents()  # clear any leftover keys

    show_shape = True

    # while loop for whole duration of audio playback
    while True:
        
        elapsed = ptb.GetSecs() 

        # record key presses continuously
        if key_pressed == False: # only record first response

            response_keys = response_kb.getKeys(key_pos, waitRelease=False)

            if response_keys and not key_pressed:
                key_pressed = True
                key_name = response_keys[0].name
                key_time = ptb.GetSecs()
                key_rt_trial = key_time - onset_soundstim # rt relative to trial onset --> compute relative to deviant onset later

            if not feedback_recorded:

                if dpos[i] != 0 and key_pressed:
                    correct_key = dpos[i] - 2
                    key_posy = key_pos.index(key_name)
                    
                    if correct_key == key_posy and key_rt_trial > time_to_dev:
                        performance[i] = 1
                        feedback_recorded = True
                        
                    elif correct_key != key_posy and key_rt_trial > time_to_dev_3:                    
                        performance[i] = 0

                        if key_rt_trial > time_to_dev_all[key_posy]:
                            feedback_recorded = True
                        elif key_rt_trial <= time_to_dev_all[key_posy]:
                            message_too_fast.text = ' '
                            feedback_recorded = True

                    elif correct_key == key_posy and key_rt_trial < time_to_dev:  
                        message_too_fast.text = ' '
                        feedback_recorded = True

                    elif key_rt_trial < time_to_dev_3:
                        message_too_fast.text = ' '
                        feedback_recorded = True   
        
        # make experiment closable by Esc key press
        if 'escape' in [k.name for k in escape_kb.getKeys(['escape'], waitRelease=False)]:
            core.quit(); sys.exit()
        
        # start by showing cue
        elif elapsed - onset <= (len_waveform[i] / sample_rate):

            if show_shape:
                if cue[i] == "circle":
                    circle.draw()
                elif cue[i] == "triangle":
                    triangle.draw()
                elif cue[i] == "square":
                    square.draw()

            # stimulus phase until tone sequence is over
            if phase == 'stimulus':
                
                if elapsed-onset <= trial_duration:
                    offset_stimulus = elapsed
               
                else:
                    phase = 'response'
                    response_start = ptb.GetSecs()
                    onset_iti = ptb.GetSecs()
                  
            # response window = small time frame after last deviant in addition to during the tone sequence
            elif phase == 'response':

                # stop shape 50 ms before comfidence rating to avoid visual overlap
                if elapsed-onset > trial_duration + response_window - 0.05:
                    show_shape = False
                    message_too_fast.text = ' '
                              
                # collect feedback (text not shown in main experiment)
                if elapsed-onset > trial_duration + response_window:

                    if key_pressed:
                        phase = 'ITI'
                    elif not key_pressed:
                        performance[i] = 3
                        feedback_recorded = True
                        phase = 'feedback'
                        feedback_start = ptb.GetSecs()

            # collect confidence ratings
            elif phase == 'feedback':
                if ptb.GetSecs() - feedback_start <= feedback_duration:
                    feed.color = feed.color
                    feed.text = feed.text
                else:
                    phase = 'ITI'

            # fixation cross for remainder of audio playback
            elif phase == 'ITI':
                message.text = '+'                           
            
            # draw and flip in each iteration of while loop
            if phase == 'stimulus':
                message_too_fast.draw()
            elif phase == 'ITI':
                message.draw()
            elif phase == 'feedback':
                message.draw()
                feed.draw()    
            elif phase == 'response':
                message_too_fast.draw()
            
            win.flip()
                    
        elif elapsed - onset > (len_waveform[i] / sample_rate):
            offset_iti = ptb.GetSecs()
            break
    
    # record come timing variables --> TODO: finalize
    offset = PsychPortAudio('Stop', pahandle, 1) # offset measured by PsychPortAudio
    offset_trial.append(offset[-1])
    onset_sound.append(onset_soundstim)
    single_onsets = [onset_soundstim + i * (stim_dur + isi_dur) for i in range(8)]
    onset_tones.append(single_onsets) # theoretical onsets based on measured trial start
    duration_sound.append(trial_duration - pre_cue) # theoretical trial duration (excluding pre_cue)
    offset_sound.append(onset + trial_duration)
    offset_sound_getsecs.append(offset_stimulus) # based on getsecs
    duration_sound_getsecs.append(offset_stimulus-onset_soundstim) # based on getsecs
    onset_iti_list_getsecs.append(onset_iti) # based on getsecs
    offset_iti_list_getsecs.append(offset_iti)
    onset_iti_list.append(onset + trial_duration) # theoretical based on measured onset + known duration
    offset_iti_list.append(onset + trial_duration + ITI_list[i]) # theoretical based on measured onset + known durations
    
    if key_pressed == True:
        key_rt_dev = key_time - single_onsets[dpos[i]] 
        keys_pressed[i] = key_name
        rts_getsecs_trial[i] = key_rt_trial
        rts_getsecs_dev[i] = key_rt_dev
    else:
        keys_pressed[i] = None
        rts_getsecs_trial[i] = None
        rts_getsecs_dev[i] = None   

    #---- present feedback trials (every 6th trial)
    if feedback_trial == True:

        trial_inds = np.where(np.array(runs) == runs[i])[0]
        count_correct = np.sum(np.array(performance)[trial_inds] == 1)
        trials_so_far = i + 1
        
        accuracy_now = (count_correct/trials_so_far)*100

        feedback_start = ptb.GetSecs()

        while ptb.GetSecs() - feedback_start <= feedback_duration + ITI_list[i]-1: # add some ITI after the feedback (bit shorter than regular ITI bc that contains response window and rating time)

            if ptb.GetSecs() - feedback_start <= feedback_duration:
                feed_acc.text = f'% korrekte Positionen in diesem Durchgang bisher: {np.abs(accuracy_now): .2f}'
                feed_acc.draw()
            else:
                feed_acc.text = '+'
                feed_acc.draw()
            
            win.flip()

#---- display overall performance and wait for end (5 s wait, then space press ends the experiment)
accuracy = ((len(np.where(np.array(performance) == 1)[0]) + len(np.where(np.array(performance) == 4)[0]))/len(performance))*100
now = ptb.GetSecs()

message.text = f"% korrekte Positionen in diesem Durchgang: {np.abs(accuracy): .2f}\n\n\nBitte gib der Versuchsleitung Bescheid!"

message.color = (1, 1, 1)
message.draw()
win.flip()

# wait for 5 s
while ptb.GetSecs() < now + 5:
    core.wait(0.1)

# end experiment by pressing space
end_kb = keyboard.Keyboard(backend = 'ptb')
end_keys = end_kb.waitKeys(keyList=['space'], waitRelease=False)

win.close()
core.quit()

