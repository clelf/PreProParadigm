#---- set preferences first --> ATTENTION: manually use ptb backend for keyboard later on anyways on MAC!
from psychopy import prefs
prefs.general['version'] = '2025.1.1'
prefs.hardware['keyboard'] = 'ptb'
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3' # could also use 4 but then no fallback in case of small deviations
# prefs.hardware['audioDevice'] = 'Externe Kopfhörer'
prefs.hardware['audioDevice'] = 'Mac mini-Lautsprecher'

#---- check psychopy version
from psychopy import useVersion
useVersion('2025.1.1')

import psychopy
print(f"Running PsychoPy {psychopy.__version__}")

#---- imports
import psychtoolbox as ptb
from psychtoolbox import audio
from psychtoolbox import PsychPortAudio

from psychopy import sound
from psychopy.sound import backend_ptb as ptb_back
from psychopy import core
from psychopy import visual
from psychopy.hardware import keyboard
from psychopy import gui
from psychopy import monitors

import pandas as pd
import numpy as np
import soundfile as sf
from datetime import date
import os

#---- create logfile directory if not already existing
os.makedirs('logfiles/', exist_ok=True)

#---- open GUI to enter participant information
trial_list_dir = 'trial_lists/' # set directory for relevant trial list files
expName = 'AuditPrePro'
expInfo = {'participant':'', 'session':''} # enter participant and session info
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,title=expName)
if dlg.OK == False:
    core.quit()
expInfo['date'] = date.today()

participant = expInfo['participant']
session = expInfo['session']

date = date.today()
date = date.strftime("%Y-%m-%d")

#---- open a screen and test if refresh rate can be measured
mon = monitors.Monitor('tempMonitor')  # name can be anything
mon.setSizePix([1280, 720])  

win = visual.Window(
    size=(1280, 720),
    fullscr=False,
    screen=1,
    units='pix',
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
    
message = visual.TextStim(win, text='Starte Experiment ...')
message.draw()
win.flip()

#---- define variables to record
onset_sound = []
onset_tones = []
duration_sound = []
offset_trial = []
offset_sound = []
onset_iti_list = []
offset_iti_list = []
rts = [] # check RT implementation!
offset_sound_imprecise = []
duration_sound_imprecise = []
onset_iti_list_imprecise = []
offset_iti_list_imprecise = []
ITI_list = []
frequency = []
keys_pressed = []
rts_imprecise = [] # check
dpos = []
performance = []
runs = []
rule = []
tone_type = []
trial_nr = []
tau = []
lim_std = []
lim_dev = []
confidence = []

# initialize all keyboards
escape_kb = keyboard.Keyboard(backend='ptb') # set keyboard than can be used to stop the experiment
escape_kb.waitKeys(maxWait=1) 

trigger_kb = keyboard.Keyboard(backend = 'ptb') # trigger kb
trigger_kb.waitKeys(maxWait=1)  

response_kb = keyboard.Keyboard(backend = 'ptb') # trigger kb
response_kb.waitKeys(maxWait=1)  

# slider + text
slider = visual.Slider(
    win=win,
    pos=(0, 0),
    size=(800, 50),
    labels=["0", "25", "50", "75", "100"],
    ticks=[0, 25, 50, 75, 100], 
    granularity=1,
    style='rating',
    color='DarkSlateBlue',
    markerColor='Red'
)
question = visual.TextStim(
    win=win,
    text="Wie sicher sind Sie sich bei Ihrer Antwort?",
    pos=(0, 150),
    color="black",
    height=30
)

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
    "1/n": [(n, 1/n) for n in range(1, 6)],
    "1/n^2": [(n, 1/(n**2)) for n in range(1, 6)],
    "e^(-0.5(n-1))": [(n, np.exp(-0.5*(n - 1))) for n in range(1, 6)],
    "odd 1/n": [(n, 1/n) for n in range(1, 10, 2)]
}

harmonics = harmonics_dict['1/n'] # choose implementation of harmonics

sample_rate = 48000 # adjust to final device used!
ramp_time = 0.01
stim_dur = 0.1
isi_dur = 0.65

response_window = 1.5 # --> too short?
trial_duration = (8*stim_dur) + (7*isi_dur)
key_pos = ['v','y','u','i','l']
# key_pos = ['4','1!','2"','3§','4$'] # something like this for 7T scanner
feedback_duration = 1 #1 is better than 0.5

#---- read in the pre-created trial lists
trials = pd.read_csv(f'{trial_list_dir}sub-{participant}/sub-{participant}_ses-{session}_trials.csv')
#trials = trials.iloc[:8] # trials for testing
n_trials = len(trials)/8

#---- more important variables
run_step = len(trials) // 4
run_indices = [(i * run_step, (i + 1) * run_step) for i in range(4)]

sound_list = []
len_waveform = []
buffer_handles = []

#---- open audio port
pahandle = PsychPortAudio('Open', [], 1, 4, sample_rate, 1) # for mac mini --> adjust to final machine!

#---- load experiment: create all pre-definable variables and load sound waves in buffer
for i in range(0, len(trials), 8):
    
    message = visual.TextStim(win, text='Lade Stimuli ...')
    message.draw()
    win.flip()
    
    waveform = []
    oddball_trial = trials['observation'][i:i + 8]
    
    dpos_trial = trials['dpos'][i:i+8]
    dposy = dpos_trial.iloc[0]
    dpos.append(dposy)
    
    run_trial = trials['run_n'][i:i+8]
    runy = run_trial.iloc[0]
    runs.append(runy)
    
    rule_trial = trials['rule'][i:i+8]
    ruly = rule_trial.iloc[0]
    rule.append(ruly)
    
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

#---- wait for trigger to start scan
message = visual.TextStim(win, text='Warten auf Scanner ...')
message.draw()
win.flip()

trigger_list = ['t'] # ['5'] for 7T
trigger_times = []
n_triggers_to_wait = 5 # how many triggers to wait for

# refine this to needs --> here comparisons just for timing verification
ptb_start = ptb.GetSecs()  # PTB global clock start
trigger_kb.clock.reset()

for t in range(n_triggers_to_wait): # start on the 5th trigger

    trigger_keys = trigger_kb.waitKeys(keyList=trigger_list, waitRelease=False)
    key_ptb_time = ptb.GetSecs()
    # t_abs_event = trigger_keys[0].tDown
    key_rt_kb = trigger_keys[0].rt  # PsychoPy's time since clock reset
    key_rt_clock = trigger_kb.clock.getTime()  # Should match key_rt_kb

    # Compare with PTB time
    diff = key_ptb_time - ptb_start - key_rt_kb
    print(f"key.rt:         {key_rt_kb:.6f} s")
    print(f"clock.getTime:  {key_rt_clock:.6f} s")
    print(f"PTB timestamp:  {key_ptb_time - ptb_start:.6f} s")
    # print(f"tDown timestamp:  {t_abs_event:.6f} s")
    print(f"Difference:     {diff:.6f} s")
    trigger_times.append(key_ptb_time)
    print(trigger_times)
  
#---- show fixation cross
message.text = '+'
message.draw()
win.flip()

#---- play oddball sequences and record logfiles separately for each run
for i in range(0, int(n_trials) + 1):
      
    if i == n_trials:
        runs.append(5)
    
    if i > 0 and (runs[i] != runs[i-1] or i == n_trials):
        start_wait = ptb.GetSecs()
        
        if i < n_trials:
            trial_run = run_indices[runs[i-1]]
            trial_run_slow = (np.array(trial_run)/8).astype(int)
        if i == n_trials:
            trial_run = run_indices[runs[i-2]]
            trial_run_slow = (np.array(trial_run)/8).astype(int)                

        accuracy_run = ((len(np.where(np.array(performance[trial_run_slow[0]:trial_run_slow[1]]) == 1)[0]) + len(np.where(np.array(performance[trial_run_slow[0]:trial_run_slow[1]]) == 4)[0]))/len(performance[trial_run_slow[0]:trial_run_slow[1]]))*100
        message.text = f"% korrekte Antworten in diesem Durchgang: {accuracy_run: .2f}\n\n\nKurze Pause!"
        message.draw()
        win.flip()
        
        data = []
        
        # write datafile (to do: make this more BIDS-like & remove unnecessary stuff!)
        data = pd.DataFrame({
            'onset': np.repeat(onset_sound[trial_run_slow[0]:trial_run_slow[1]],8),
            'duration': np.repeat(duration_sound[trial_run_slow[0]:trial_run_slow[1]],8),
            'trial_type': np.concatenate(tone_type)[trial_run[0]:trial_run[1]],
            'onset_tone': np.concatenate(onset_tones)[trial_run[0]:trial_run[1]],
            'duration_imprecise': np.repeat(duration_sound_imprecise[trial_run_slow[0]:trial_run_slow[1]],8),
            'response_time_imprecise': np.repeat(rts_imprecise[trial_run_slow[0]:trial_run_slow[1]],8),
            'offset_trial': np.repeat(offset_trial[trial_run_slow[0]:trial_run_slow[1]],8),
            'offset_sound_imprecise': np.repeat(offset_sound_imprecise[trial_run_slow[0]:trial_run_slow[1]],8),
            'onset_iti': np.repeat(onset_iti_list[trial_run_slow[0]:trial_run_slow[1]],8),
            'offset_iti': np.repeat(offset_iti_list[trial_run_slow[0]:trial_run_slow[1]],8),
            'onset_iti_imprecise': np.repeat(onset_iti_list_imprecise[trial_run_slow[0]:trial_run_slow[1]],8),
            'offset_iti_imprecise': np.repeat(offset_iti_list_imprecise[trial_run_slow[0]:trial_run_slow[1]],8),
            'ITI': np.repeat(ITI_list,8)[trial_run[0]:trial_run[1]],
            'frequency': frequency[trial_run[0]:trial_run[1]],
            'key_pressed': np.repeat(keys_pressed[trial_run_slow[0]:trial_run_slow[1]],8),
            'confidence': np.repeat(confidence[trial_run_slow[0]:trial_run_slow[1]],8),
            'correct': np.repeat(performance[trial_run_slow[0]:trial_run_slow[1]],8),
            'dpos': np.repeat(dpos,8)[trial_run[0]:trial_run[1]],
            'rule': np.repeat(rule,8)[trial_run[0]:trial_run[1]],
            'runs': np.repeat(runs,8)[trial_run[0]:trial_run[1]],
            'trial_no': np.repeat(trial_nr,8)[trial_run[0]:trial_run[1]],
            'tau_std': np.repeat(tau,8)[trial_run[0]:trial_run[1]],
            'lim_std': np.repeat(lim_std,8)[trial_run[0]:trial_run[1]],
            'lim_dev': np.repeat(lim_dev,8)[trial_run[0]:trial_run[1]],
            'trigger_time_1': np.repeat(trigger_times[0], len(frequency[trial_run[0]:trial_run[1]])),
            'trigger_time_2': np.repeat(trigger_times[1], len(frequency[trial_run[0]:trial_run[1]])),
            'trigger_time_3': np.repeat(trigger_times[2], len(frequency[trial_run[0]:trial_run[1]])),
            'trigger_time_4': np.repeat(trigger_times[3], len(frequency[trial_run[0]:trial_run[1]])),
            'trigger_time_5': np.repeat(trigger_times[4], len(frequency[trial_run[0]:trial_run[1]])),
        })
        
        if runs[i] == 5:
             data.to_csv(f'logfiles/sub-{participant}-ses-{session}-run-{str(runs[i]-1).zfill(2)}-events_{date}.tsv', sep='\t', index=False)
        else:
            data.to_csv(f'logfiles/sub-{participant}-ses-{session}-run-{str(runs[i]).zfill(2)}-events_{date}.tsv', sep='\t', index=False)
        
            # wait for key press to initialize wait for trigger
            print('Press space to continue with next run...')
            pause_kb = keyboard.Keyboard(backend = 'ptb')
            pause_keys = pause_kb.waitKeys(keyList=['space'], waitRelease=False)
             
        if runs[i] != 5:
            # then wait for trigger to start off again
            message = visual.TextStim(win, text='Warten auf Scanner ...')
            message.draw()
            win.flip()
            
            trigger_times = []
            # refine this to needs --> here comparisons just for timing verification
            trigger_kb = keyboard.Keyboard(backend = 'ptb') # important to set! otherwise overwritten for some reason
            ptb_start = ptb.GetSecs()  # PTB global clock start
            trigger_kb.clock.reset()

            for t in range(n_triggers_to_wait): # start on the 5th trigger

                trigger_keys = trigger_kb.waitKeys(keyList=trigger_list, waitRelease=False)
                key_ptb_time = ptb.GetSecs()
                key_rt_kb = trigger_keys[0].rt  # PsychoPy's time since clock reset
                key_rt_clock = trigger_kb.clock.getTime()  # Should match key_rt_kb

                # Compare with PTB time
                diff = key_ptb_time - ptb_start - key_rt_kb
                print(f"key.rt:         {key_rt_kb:.6f} s")
                print(f"clock.getTime:  {key_rt_clock:.6f} s")
                print(f"PTB timestamp:  {key_ptb_time - ptb_start:.6f} s")
                print(f"Difference:     {diff:.6f} s")
                trigger_times.append(key_ptb_time)
                print(trigger_times)
                   
            message.text = '+'
            message.draw()
            win.flip()
            
        else:
            break
        
    # play audio sequences from buffer
    PsychPortAudio('UseSchedule', pahandle, 1)  # 1 = replace current schedule
    PsychPortAudio('AddToSchedule', pahandle, buffer_handles[i])
    onset = PsychPortAudio('Start', pahandle, 1, 0, 1)
        
    key_pressed = False
    feedback = False
    slider_rating = None
    slider_start = None
    slider_end = None
    slider_time = 0
    max_slider_time = 4
    slider.reset()
    #response_kb = keyboard.Keyboard(backend = 'ptb')
    message_color = (1, 1, 1)
    
    while True:
        
        elapsed = ptb.GetSecs() 
        
        # make experiment closable
        escape_keys = escape_kb.getKeys(keyList=['escape'], waitRelease=False)
        if 'escape' in [k.name for k in escape_keys]:
            core.quit()
            sys.exit()

        if elapsed-onset <= (len_waveform[i] / sample_rate):
            
            if elapsed-onset <= trial_duration:
                offset_stimulus = elapsed
                onset_iti = ptb.GetSecs()
                
            offset_iti = elapsed
              
            if elapsed - onset > trial_duration and elapsed - onset <= (trial_duration + response_window):
                
                message_color = (0, 0, 1)
                response_keys = response_kb.getKeys(key_pos, waitRelease=False)
                
                if response_keys and not key_pressed: # record only first response --> change?
                    key_time = ptb.GetSecs()
                    key_rt = key_time - offset_stimulus
                    key_name = response_keys[0].name
                    key_pressed = True
                    
                    slider_start = ptb.GetSecs()

                    while ptb.GetSecs() - slider_start <= max_slider_time and slider_rating is None:
                        question.draw()
                        slider.draw()
                        win.flip()

                        if slider.getRating() is not None:
                            slider_rating = slider.getRating()
                            slider_end = ptb.GetSecs()
                            slider_time = slider_end - slider_start
                            confidence.append(slider_rating)

                    # if time ran out with no rating, still record
                    if slider_rating is None:
                        slider_rating = slider.getRating()
                        slider_end = ptb.GetSecs()
                        slider_time = slider_end - slider_start
                        confidence.append(slider_rating)
                        
                    # add whether pressed key is correct or not
                    if dpos[i] != 0:                     
                         correct_key = dpos[i] - 2
                         key_posy = key_pos.index(key_name)
                         
                         if correct_key == key_posy:
                             performance.append(1)
                             # add visual feedback (turn fixation green) --> duration?
                             feedback_color = (0, 1, 0)
                         else:
                             performance.append(0)
                             # add visual feedback (turn fixation red) --> duration?
                             feedback_color = (1, 0, 0)
                    else:
                         performance.append(2) # false alarm in omission trials
                         feedback_color = (1, 0, 0)
                    feedback = True
                    feedback_start = ptb.GetSecs()

            if feedback:
                if ptb.GetSecs() - feedback_start <= feedback_duration:
                    message_color = feedback_color
                else:
                    feedback = False  # feedback expired
                    message_color = (0, 0, 1) # back to blue
                 
            # draw fixation/message and flip
            message.color = message_color
            message.draw()
            win.flip()
                    
        elif elapsed-onset > (len_waveform[i] / sample_rate):
            message.color = (1, 1, 1)
            message.draw()
            win.flip()
            # add if response is missing
            if not key_pressed and dpos[i] != 0:
                performance.append(3) # miss
            if not key_pressed and dpos[i] == 0:
                performance.append(4) # correct omission of response in trials w/o deviant
            break
    
    # record come timing variables --> finalize
    offset = PsychPortAudio('Stop', pahandle, 1)
    offset_trial.append(offset[-1])
    onset_sound.append(onset)
    single_onsets = [onset + i * (stim_dur + isi_dur) for i in range(8)]
    onset_tones.append(single_onsets)
    duration_sound.append(trial_duration)
    offset_sound_imprecise.append(offset_stimulus)
    duration_sound_imprecise.append(offset_stimulus-onset)
    onset_iti_list_imprecise.append(onset_iti)
    offset_iti_list_imprecise.append(offset_iti)
    onset_iti_list.append(onset + trial_duration)
    offset_iti_list.append(onset + trial_duration + ITI_list[i])
    
    if key_pressed == True:
        keys_pressed.append(key_name)
        rts_imprecise.append(key_rt)
    else:
        keys_pressed.append(None)
        rts_imprecise.append(np.nan)
        confidence.append(np.nan)
    

#---- display overall performance and wait for end (5 s wait, then space press ends the experiment)
accuracy = ((len(np.where(np.array(performance) == 1)[0]) + len(np.where(np.array(performance) == 4)[0]))/len(performance))*100
now = ptb.GetSecs()
message.text = f"% korrekte Antworten insgesamt: {accuracy: .2f}\n\n\nWarten auf Ende des Experiments!"
message.color = (1, 1, 1)
message.draw()
win.flip()

while ptb.GetSecs() < now + 5:
    core.wait(0.1)

end_kb = keyboard.Keyboard(backend = 'ptb')
end_keys = end_kb.waitKeys(keyList=['space'], waitRelease=False)

win.close()
core.quit()

