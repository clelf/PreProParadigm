# ----------------------------------------------------------#
# CHECK PSYCHOPY VERSION
# ----------------------------------------------------------#
import psychopy
print(f"Running PsychoPy {psychopy.__version__}")

# ----------------------------------------------------------#
# SET PSYCHOPY PREFS
# ----------------------------------------------------------#
from psychopy import prefs
prefs.hardware['keyboard'] = 'ptb'
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'

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

from psychopy import core
from psychopy import visual
from psychopy.hardware import keyboard
from psychopy import gui
from psychopy import monitors

import pandas as pd
import numpy as np

from scipy.io.wavfile import write
from scipy.interpolate import PchipInterpolator

from datetime import date, datetime
import os
import sys

# ----------------------------------------------------------#
# FUNCTIONS
# ----------------------------------------------------------#
def generate_isi(duration, sample_rate):

    '''generates interval sound (array of zeros)
    
    Args:
        - duration: duration in s
        - sample_rate: sampling rate in Hz

    Returns:
        - waveform_isi: array constituting silent interval
    '''
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform_isi = np.zeros_like(t)

    return waveform_isi

def generate_hct(f0, duration, sample_rate, ramp_time, k, a):

    '''generates harmonic complex tone
    
    Args:
        - f0: fundamental frequency in Hz
        - duration: duration in seconds
        - sample_rate: sampling rate in Hz
        - ramp_time: hanning window time in seconds
        - k: number of harmonics
        - a: decay factor for harmonic amplitudes

    Returns:
        - hct: array corresponding to harmonic complex tone    
    '''

    harmonics = [(k, 1 / (k ** a)) for k in range(1, k + 1)]
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    for multiple, amplitude in harmonics:
        waveform += amplitude * np.sin(2 * np.pi * f0 * multiple * t)
    
    ramp_samples = int(sample_rate * ramp_time)
    full_window = np.hanning(2 * ramp_samples)
    ramp_in = full_window[:ramp_samples]
    ramp_out = full_window[ramp_samples:]

    envelope = np.ones_like(waveform)
    envelope[:ramp_samples] *= ramp_in
    envelope[-ramp_samples:] *= ramp_out

    hct = waveform * envelope 

    return hct

def load_interpolator(path, column = 'weights'):
    with open(path) as f:
        header = f.readline().strip().split(',')
    data = np.loadtxt(path, delimiter = ',', skiprows = 1)
    f0s = data[:,0]
    values = data[:,header.index(column)]
    return PchipInterpolator(np.log(f0s), values)    

# ----------------------------------------------------------#
# SETUP
# ----------------------------------------------------------#
config = {
    'exp_name': 'AuditPreProTraining',
    'screen_size': [3840, 2160],
    'screen_units': 'pix',
    'mouse_visible': False,
    'log_dir': 'logfiles_training/',
    'trial_dir': 'trial_lists_training/',
    'trigger_key': 'space',
    'N_triggers_wait': 1,
    'sample_rate': 48000,
    'ISI': 0.65,
    'duration': 0.1,
    'ramp_time': 0.01,
    'k': 5,
    'a': 1,
    'key_pos': ['v','y','u','i','l'],
    'key_pos_catch': ['v','l'],
    'feedback_duration': 2,
    'response_window': 0.65,
    'correct_after_dev': 1.5,
    'catch_duration': 3,
    'catch_feedback_duration': 2,
    'pre_cue': 0.75,
    'cuesize': 50,
    'n_tones': 8,
    'n_runs': 4,
    'n_catch': 10
}

trial_duration = (config['n_tones']*config['duration']) + ((config['n_tones']-1)*config['ISI']) + config['pre_cue']

os.makedirs(f"{config['log_dir']}", exist_ok=True)

#---- open GUI to enter participant information
expName =  config['exp_name']
expInfo = {'participant':'', 'session':'', 'run': ''}

dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,title=expName)
if dlg.OK == False:
    core.quit()
expInfo['date'] = date.today()

participant = expInfo['participant']
session = expInfo['session']
run_info = int(expInfo['run'])

date = datetime.now().strftime("%Y-%M-%d_%H_%M-%S")

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
trigger_kb = keyboard.Keyboard(backend = 'ptb')
response_kb = keyboard.Keyboard(backend = 'ptb')
catch_kb = keyboard.Keyboard(backend='ptb')
pause_kb = keyboard.Keyboard(backend = 'ptb')
end_kb = keyboard.Keyboard(backend = 'ptb')

# ----------------------------------------------------------#
# SETUP MESSAGES
# ----------------------------------------------------------#
message_too_fast = visual.TextStim(win, text='', wrapWidth=2000)
message_too_fast.height = 50
message_too_fast.color = (1,0,0)
message_too_fast.pos = (0, 150)

#---- feedback for too slow
feed = visual.TextStim(win, text='zu langsam', wrapWidth=2500)
feed.color = (1, 0, 0)
feed.height = 50
feed.pos = (0, 150)

# feedback for accuracy in last run
feed_acc = visual.TextStim(win, text=' ', wrapWidth=2500)
feed_acc.color = (1, 1, 1)
feed_acc.height = 100
feed_acc.pos = (0, 0)

catch_message = visual.TextStim(win, text='Welche Regel war im letzten Durchgang aktiv?\n\n\n\nRegel 1                 Regel 2', wrapWidth=2000)
catch_message.color = (1, 1, 1)
catch_message.height = 100
catch_message.pos = (0,0)

# ----------------------------------------------------------#
# SETUP CUE SHAPES
# ----------------------------------------------------------#
cuesize = config['cuesize']

shapes = {
    'pentagon': visual.Polygon(
        win,
        edges=5,
        radius=cuesize,
        fillColor='white',
        lineColor=None
    ),
    'cross': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize/2, -cuesize),
            ( cuesize/2, -cuesize),
            ( cuesize/2, -cuesize/2),
            ( cuesize,   -cuesize/2),
            ( cuesize,    cuesize/2),
            ( cuesize/2,  cuesize/2),
            ( cuesize/2,  cuesize),
            (-cuesize/2,  cuesize),
            (-cuesize/2,  cuesize/2),
            (-cuesize,    cuesize/2),
            (-cuesize,   -cuesize/2),
            (-cuesize/2, -cuesize/2)
        ],
        fillColor='white',
        lineColor=None
        )
    }

# ----------------------------------------------------------#
# INITIALIZE VARS
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Starte Experiment ...', wrapWidth=2000)
message.height = 100
message.draw()
win.flip()

#---- define variables to record
onset_sound = []
onset_tones = []

onset_cues = []
offset_cues = []

duration_sound = [] # based on getsecs

ITI_list = [] # theoretical ITIs

tau = [] # taus
frequency = [] # tone frequency in Hz
lim_std = [] # mu std
lim_dev = [] # mus dev

rule = [] # rule
cue = [] # cue
dpos = [] # deviant positions
tone_type = [] # std or deviant

runs = [] # run no.
trial_nr = [] # trial no.

catch_all = []
catch_key_list = []
catch_correct_list = []
catch_rt_list = []
is_catch = []

feedback_trialy = []

# ----------------------------------------------------------#
# LOAD STIMULI
# ----------------------------------------------------------#
trials = pd.read_csv(f"{config['trial_dir']}sub-all/sub-all_ses-{session}_trials.csv")

trials = trials[trials['run_n'] == run_info-1]
n_trials = len(trials)/config['n_tones']

rts_getsecs_trial = [None]*int(n_trials) # RT based on ptb.getSecs() from onset trial
rts_getsecs_dev = [None]*int(n_trials) # RT based on ptb.getSecs() from onset deviant
keys_pressed = [None]*int(n_trials) # pressed key
performance = [None]*int(n_trials) # correct?

#---- more important variable
run_step = len(trials) // config['n_runs']# n trials per run
run_indices = [(i * run_step, (i + 1) * run_step) for i in range(config['n_runs'])] # trials indices where run changes

len_waveform = [] # length waveform (used to compute playback duration)
buffer_handles = [] # sound buffer

# create all sound stimuli (incl. max amplitude weighing and loudness normalization)
weight_list = pd.read_csv("weights_n_harmonics_5_a_1_duration_0.1_ramp_time_0.01_target_sone_1.csv")

sound_raw = [[] for _ in range(len(pd.unique(trials["trial_n"])))]
sound_norm = [[] for _ in range(len(pd.unique(trials["trial_n"])))]
f0s_all = [[] for _ in range(len(pd.unique(trials["trial_n"])))]

for i in pd.unique(trials["trial_n"]):
    freq_data = trials[trials["trial_n"] == i]
    offsety = trials["trial_n"].iloc[0]
    for s in freq_data["frequency"]:

        f0s_all[i-offsety].append(s)

        raw = generate_hct(
            round(s, 2),
            config['duration'],
            config['sample_rate'],
            config['ramp_time'],
            config['k'],
            config['a']
        )

        sound_raw[i-offsety].append(raw)

        norm = raw / weight_list['max_amp'][0]
        sound_norm[i-offsety].append(norm)


# loudness equalization!!! only works if the CSV file containing the weight was created with the exact tone duration and harmonic configuration as in config here!!!
sound_loud_weight = [[] for _ in range(len(pd.unique(trials["trial_n"])))] 
interpolator = load_interpolator("weights_n_harmonics_5_a_1_duration_0.1_ramp_time_0.01_target_sone_1.csv", 'weights')

for i in pd.unique(trials["trial_n"]):
    freq_data = trials[trials["trial_n"] == i]
    offsety = trials["trial_n"].iloc[0]
    for n,s in enumerate(freq_data["frequency"]):
        freq = f0s_all[i-offsety][n]
        weight = float(interpolator(np.log(freq)))
        weighted = sound_norm[i-offsety][n] * weight
        sound_loud_weight[i-offsety].append(weighted)

#---- open audio port
pahandle = PsychPortAudio('Open', [], 1, 4, config['sample_rate'], 1) # TODO sometimes necessary to adjust to device

#---- load experiment: create all trials in advance from previously generated trial_list
for i in pd.unique(trials["trial_n"]):
    offsety = trials["trial_n"].iloc[0]
    
    message = visual.TextStim(win, text='Lade Stimuli ...', wrapWidth=2000)
    message.height = 100
    message.draw()
    win.flip()

    temp = trials[trials["trial_n"] == i]
    
    waveform = []
    oddball_trial = temp['frequency']
    
    dpos_trial = temp['dpos']
    dposy = dpos_trial.iloc[0]
    
    if dposy is not None:
        dposy = int(dposy)
    dpos.append(dposy)
    
    run_trial = temp['run_n']
    runy = run_trial.iloc[0]
    runs.append(runy)
    
    rule_trial = temp['rule']
    ruly = rule_trial.iloc[0]
    rule.append(ruly)

    cue_trial = temp['cue']
    cuey = cue_trial.iloc[0]
    cue.append(cuey)
    
    tones_trial = np.zeros(config['n_tones'])
    if dposy is not None:
        tones_trial[dposy] = 1
    tone_type.append(tones_trial.astype(int))
    
    trial_trial = temp['trial_n']
    trialy = trial_trial.iloc[0]
    trial_nr.append(trialy)
    
    tau_std_trial = temp['tau_std']
    tauy = tau_std_trial.iloc[0]
    tau.append(tauy)
    
    lim_std_trial = temp['lim_std']
    lim_stdy = lim_std_trial.iloc[0]
    lim_std.append(lim_stdy)
    
    lim_dev_trial = temp['lim_dev']
    lim_devy = lim_dev_trial.iloc[0]
    lim_dev.append(lim_devy)
            
    ITI = temp['ITI'].iloc[0]
    ITI_list.append(ITI)    
   
    tone_count = 0

    wave = generate_isi(config['pre_cue'], config['sample_rate']) # generate silence for pre-cueing
    waveform.append(wave)
    
    for n,s in enumerate(oddball_trial):

        tone_count += 1
        frequency.append(round(s,2))
        #wave = generate_hct(round(s,2), config['duration'], config['sample_rate'], config['ramp_time'], config['k'], config['a'])
        waveform.append(sound_loud_weight[i-offsety][n])
        
        if tone_count < config['n_tones']:
            isi = generate_isi(config['ISI'], config['sample_rate'])
            waveform.append(isi)
        elif tone_count == config['n_tones']:
            iti_wave = generate_isi(ITI, config['sample_rate'])
            waveform.append(iti_wave)
        
    waveform = np.concatenate(waveform)
    len_waveform.append(len(waveform))
    
    waveform = waveform.astype(np.float32)
    buffer_handle = PsychPortAudio('CreateBuffer', [], waveform)
    buffer_handles.append(buffer_handle)

#---- insert catch trials
catch_pos = [[]]*len(np.unique(runs))
run_count = -1

for runn in np.unique(runs):

    run_count += 1

    done = False

    positions = np.where(
    (np.array(runs) == runn) &
    (np.isin(np.array(rule), [0, 1]))
    )
    
    # get indices for dev5 vs != dev5
    positions_5 = positions[0][np.where((np.array(dpos)[positions[0]] == 4))[0]]
    positions_not_5 = positions[0][np.where((np.array(dpos)[positions[0]] != 4))[0]]

    while done == False:
        catch_pos[run_count] = []
        
        # make sure that 50% of catch trials are after dev5
        while len(catch_pos[run_count]) < config['n_catch']/2:
            test_pos = np.random.choice(positions_5)
            catch_pos[run_count].append(test_pos)
        while len(catch_pos[run_count]) < config['n_catch']:
            test_pos = np.random.choice(positions_not_5)
            catch_pos[run_count].append(test_pos)

        catch_pos[run_count] = sorted(catch_pos[run_count])

        done = (
        all(catch_pos[run_count][i] - catch_pos[run_count][i-1] > 2 for i in range(1, len(catch_pos[run_count]))) # at least two trials in between catch trials
        and not np.isin(catch_pos[run_count], [np.where(runs == runn)[0][0], np.where(runs == runn)[0][-1]]).any() # catch trial not first or last trial of a given run
        )

    data_run = np.array([runs == runn])
    catch_pos_run = np.zeros(len(np.where(data_run[0]==True)[0]))
    catch_pos[run_count] = list(map(lambda x: x + 1, catch_pos[run_count]))
    catch_pos_run[catch_pos[run_count]-np.where(runs == runn)[0][0]] = 1

    catch_all.append(catch_pos_run)

catch_all = [item for sublist in catch_all for item in sublist] 

# ----------------------------------------------------------#
# WAIT FOR TRIGGER
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Drücke die Leertaste, um zu starten.', wrapWidth=2000)
message.height = 100
message.draw()
win.flip()

trigger_key = [config['trigger_key']]
trigger_times = []

for t in range(config['N_triggers_wait']): 
    trigger_keys = trigger_kb.waitKeys(keyList=trigger_key, waitRelease=False)
    trigger_time = ptb.GetSecs()
    trigger_times.append(trigger_time)
  
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

# ----------------------------------------------------------#
# RUN EXPERIMENT
# ----------------------------------------------------------#
feedback_trial = False

for i in range(0, int(n_trials) + 1):

    #print(np.sum(np.array(performance) == 'correct'))

    if runs[0] == (config['n_runs']-1):
        # get positions of feedback trials (in every 6th trial); not in last trial and only in last run
        if i == 0 or i == n_trials:
            feedback_trial = False
            feedback_trialy.append(0)
        if (i+1) % 6 == 0 and i > 0 and i <= n_trials - 1:
            feedback_trial = True
            feedback_trialy.append(1)
        elif (i+1) % 6 != 0 and i > 0 and i <= n_trials - 1:
            feedback_trial = False
            feedback_trialy.append(0)
    else:  
        if i > 0 and i <= n_trials:
            feedback_trial = False
            feedback_trialy.append(0)      
    
    # whenever run changes:
    if i == n_trials:

        data = []    

        #print(len(onset_sound), len(onset_tones), len(duration_sound), 
        #      len(ITI_list), len(rts_getsecs_trial), len(rts_getsecs_dev), len(tau), len(frequency), len(lim_std), len(lim_dev),len(keys_pressed),
        #      len(performance), len(rule),len(dpos),len(cue),len(tone_type),len(runs),len(trial_nr),len(feedback_trialy)) 

        # ----------------------------------------------------------#
        # WRITE DATA OF PREVIOUS RUN
        # ----------------------------------------------------------#
        data = pd.DataFrame({
            'onset': np.repeat(onset_sound,config['n_tones']),
            'duration': np.repeat(duration_sound,config['n_tones']),
            'trial_type': np.concatenate(tone_type),
            'response_time': np.repeat(rts_getsecs_trial,config['n_tones']),
            'onset_tone': np.concatenate(onset_tones),
            'f0_tone': frequency,
            'response_time_dev': np.repeat(rts_getsecs_dev,config['n_tones']),
            'key_pressed': np.repeat(keys_pressed,config['n_tones']), 
            'correct': np.repeat(performance,config['n_tones']), 
            'rule': np.repeat(rule,config['n_tones']),
            'cue': np.repeat(cue,config['n_tones']),
            'dpos': np.repeat(dpos,config['n_tones']),
            'is_catch_prev': np.repeat(is_catch,config['n_tones']),
            'catch_key': np.repeat(catch_key_list,config['n_tones']),
            'catch_correct': np.repeat(catch_correct_list,config['n_tones']),
            'catch_rt': np.repeat(catch_rt_list,config['n_tones']),
            'feedback_trial': np.repeat(feedback_trialy,config['n_tones']),
            'ITI': trials['ITI'],
            'tau_std': np.repeat(tau,config['n_tones']),
            'lim_std': np.repeat(lim_std,config['n_tones']),
            'lim_dev': np.repeat(lim_dev,config['n_tones']), 
            'trial': np.repeat(trial_nr,config['n_tones']),
            'run': np.repeat(runs,config['n_tones']),
            'onset_cues':  np.repeat(onset_cues,config['n_tones']),
            'offset_cues':  np.repeat(offset_cues,config['n_tones']),
            })
        
        data.to_csv(f"{config['log_dir']}/sub-{participant}_ses-{session}_task-AuditPreProTraining_run-{run_info}_events_{date}.tsv", sep='\t', index=False)
        break

    # ----------------------------------------------------------#
    # IMPLEMENT CATCH TRIALS
    # ----------------------------------------------------------#
    if catch_all[i] == 1 and runs[i]!= config['n_runs']-1: # in all but last run, we will have catch trials
        
        catch_correct = None
        catch_feedback = False
        catch_press = False
        catch_name = None
        catch_rt_trial = None
        catch_start = ptb.GetSecs()

        catch_message.text = 'Welche Regel war im letzten Durchgang aktiv?\n\n\n\nRegel 1                 Regel 2'

        while (ptb.GetSecs() - catch_start) <= (config['catch_duration'] + 4):

            #---- present catch message
            catch_now = ptb.GetSecs()

            if 'escape' in [k.name for k in escape_kb.getKeys(['escape'], waitRelease=False)]:
                core.quit(); sys.exit()
            
            rule_correct = rule[i-1]
            catch_keys = catch_kb.getKeys(config['key_pos_catch'], waitRelease=False)

            #---- if key pressed get accuracy
            if not catch_press and catch_keys and (catch_now - catch_start <= config['catch_duration']): # upon the first key press

                catch_press = True

                catch_name = catch_keys[0].name
                catch_time = ptb.GetSecs()
                catch_rt_trial = catch_time - catch_start

                # check accuracy and change color accordingly
                if catch_name == config['key_pos_catch'][0] and rule_correct == 0:
                    catch_correct = 1
                    catch_message.text = 'richtig'
                    catch_message.color = (0,1,0)
                elif catch_name == config['key_pos_catch'][1] and rule_correct == 1:
                    catch_correct = 1
                    catch_message.text = 'richtig'
                    catch_message.color = (0,1,0)
                else:
                    catch_correct = 0
                    catch_message.text = 'falsch'
                    catch_message.color = (1,0,0) 

                catch_feedback = True                          

            #---- provide feedback for 2 seconds and go back to fixation
            if catch_feedback:
                    
                if catch_now - catch_time <= config['catch_feedback_duration'] and runs[i] != config['n_runs']-1:
                    pass
                else:
                    catch_message.text = '+'
                    catch_message.color = (1,1,1)
                
            #---- if no response after catch duration; feedback = too slow and go back to fixation
            elif not catch_feedback and (catch_now-catch_start) > config['catch_duration']:  
                if (catch_now-catch_start) <= config['catch_duration'] + 2:
                    catch_message.text = 'zu langsam'
                    catch_message.color = (1,0,0)
                    catch_rt_trial = None
                    catch_correct = 0
                    catch_name = None
                else:
                    catch_message.text = '+'
                    catch_message.color = (1,1,1)
                
            catch_message.draw()
            win.flip()    
        
        #---- collect catch trial data
        catch_key_list.append(catch_name)
        catch_correct_list.append(catch_correct)
        catch_rt_list.append(catch_rt_trial)  
        is_catch.append(1)   

    #---- if no catch trial, append empty data to match length of logfile
    elif catch_all[i] == 0 or runs[i] == config['n_runs']-1:
        catch_key_list.append(None)
        catch_correct_list.append(None)
        catch_rt_list.append(None)  
        is_catch.append(0)         

    # ----------------------------------------------------------#
    # REGULAR TRIALS
    # ----------------------------------------------------------#
    phase ='stimulus'
    key_pressed = False
    feedback_recorded = False
    response_start = None
    key_name = None
    key_rt_trial = None
    key_rt_dev = None
    cue_start = None
    cue_end = None

    #---- start playing audio sequences from buffer
    PsychPortAudio('UseSchedule', pahandle, 1)  # 1 = replace current audio schedule
    PsychPortAudio('AddToSchedule', pahandle, buffer_handles[i]) # schedule
    onset = PsychPortAudio('Start', pahandle, 1, 0, 1) # measure onset trial based on PsychPortAudio
    cue_start = onset - first_trigger
    onset_soundstim = onset + config['pre_cue'] # compute stimulus onset as onset + pre-cueing duration

    time_to_dev = (dpos[i]*config['duration']) + (dpos[i]*config['ISI']) # time from trial onset to start of deviant
    time_to_dev_all = [((np.unique(dpos)[j-2]*config['duration']) + (np.unique(dpos)[j-2]*config['ISI'])) for j in np.unique(dpos)] # onset of all deviant positions
    time_to_dev_3 = (2*config['duration']) + (2*config['ISI']) # onset of deviant 3
    
    response_kb.clearEvents()  # clear any leftover keys

    show_shape = True

    # while loop for trial progression
    while True and dpos[i] is not None:
        
        elapsed = ptb.GetSecs() 

        # collect responses from start of audio playback
        if key_pressed == False: # only record first response

            response_keys = response_kb.getKeys(config['key_pos'], waitRelease=False)

            if response_keys and not key_pressed:
                key_pressed = True
                key_name = response_keys[0].name
                key_time = ptb.GetSecs()
                key_rt_trial = key_time - onset_soundstim

            if not feedback_recorded: # if no feedback recorded yet

                if dpos[i] != 0 and key_pressed:
                    correct_key = dpos[i] - 2 # get correct answer
                    key_posy = config['key_pos'].index(key_name)
                    
                    if correct_key == key_posy and key_rt_trial > time_to_dev: # in case of correct answer
                        
                        if (key_rt_trial - time_to_dev_all[key_posy]) <= config['correct_after_dev']:
                            performance[i] = 'correct'
                            if runs[i] != config['n_runs']-1: # change cue color for correct feedback in all but last block
                                if cue[i] in shapes:
                                    shapes[cue[i]].fillColor = (0, 1, 0)
                            feedback_recorded = True
                            
                        elif (key_rt_trial - time_to_dev_all[key_posy]) > config['correct_after_dev']: # if RT > 1.5 seconds after deviant onset --> too slow
                            performance[i] = 'correct too slow'
                            if runs[i] != config['n_runs']-1:
                                message_too_fast.text = 'zu langsam'
                                if cue[i] in shapes:
                                    shapes[cue[i]].fillColor = (0, 1, 0)
                            feedback_recorded = True 
                        
                    elif correct_key != key_posy and key_rt_trial > time_to_dev_3: # in case of incorrect answer happening after position 3

                        if key_rt_trial > time_to_dev_all[key_posy]: # if incorrectly responded after deviant

                            if (key_rt_trial - time_to_dev_all[key_posy]) <= config['correct_after_dev']:
                                performance[i] = 'incorrect' 
                                if runs[i] != config['n_runs']-1: # if not last run, provide negative feedback
                                    if cue[i] in shapes:
                                        shapes[cue[i]].fillColor = (1, 0, 0)
                                feedback_recorded = True

                            elif (key_rt_trial - time_to_dev_all[key_posy]) > config['correct_after_dev']: # or too slow feedback if after RT window
                                performance[i] = 'incorrect too slow'
                                if runs[i] != config['n_runs']-1:
                                    message_too_fast.text = 'zu langsam'
                                    if cue[i] in shapes:
                                        shapes[cue[i]].fillColor = (1, 0, 0)
                                    feedback_recorded = True 

                        elif key_rt_trial <= time_to_dev_all[key_posy]:
                            performance[i] = 'too fast'
                            if runs[i] != config['n_runs']-1:
                                message_too_fast.text = 'zu schnell'
                            feedback_recorded = True

                    elif correct_key == key_posy and key_rt_trial < time_to_dev: # too fast if correct answer is given before the dev actually occurs
                        performance[i] = 'too fast'
                        if runs[i] != config['n_runs']-1:
                            message_too_fast.text = 'zu schnell'
                        feedback_recorded = True

                    elif key_rt_trial < time_to_dev_3: # too fast if response before tone 3
                        performance[i] = 'too fast'
                        if runs[i] != config['n_runs']-1:
                            message_too_fast.text = 'zu schnell'
                        feedback_recorded = True

        #---- make experiment closable by Esc key press at all times
        if 'escape' in [k.name for k in escape_kb.getKeys(['escape'], waitRelease=False)]:
            core.quit(); sys.exit()
        
        #---- while audio schedule is playing...
        elif elapsed - onset <= (len_waveform[i] / config['sample_rate']):

            # ----------------------------------------------------------#
            # PRESENT CUE
            # ----------------------------------------------------------#
            if show_shape:
                if cue[i] in shapes:
                    shapes[cue[i]].draw()

            # ----------------------------------------------------------#
            # PRESENT STIMULUS
            # ----------------------------------------------------------#
            if phase == 'stimulus':
                if elapsed-onset <= trial_duration:
                    offset_stimulus = elapsed
                else:
                    phase = 'response'
                    response_start = ptb.GetSecs()
                  
            # ----------------------------------------------------------#
            # SMALL RESPONSE WINDOW AFTER LAST TONE (CUE STILL PRESENTED)
            # ----------------------------------------------------------#
            elif phase == 'response':
                # remove shape cue 50 ms before fixation to avoid visual overlap
                if show_shape and (elapsed-onset) > trial_duration + config['response_window'] - 0.05:
                    show_shape = False
                    cue_end = ptb.GetSecs() - first_trigger
                    message_too_fast.text = ' '
                    
                    if cue[i] in shapes:
                        shapes[cue[i]].fillColor = 'white'

                if (elapsed-onset) > trial_duration + config['response_window']:
                    
                    if key_pressed:
                        phase = 'ITI'
                    elif not key_pressed:
                        performance[i] = 'miss'
                        feedback_recorded = True
                        phase = 'feedback'
                        feedback_start = ptb.GetSecs()

            elif phase == 'feedback':
                if ptb.GetSecs() - feedback_start <= config['feedback_duration']:
                    feed.color = feed.color
                    feed.text = feed.text
                else:
                    phase = 'ITI'

            # ----------------------------------------------------------#
            # ITI
            # ----------------------------------------------------------#
            elif phase == 'ITI':
                message.text = '+' 
                        
            # draw display text and flip window in each loop iteration
            if phase == 'ITI':
                message.draw() 
            elif phase == 'feedback':
                message.draw()
                if runs[i] != (config['n_runs']-1):
                    feed.draw()
            elif phase == 'response':
                message_too_fast.draw()
            elif phase == 'stimulus':
                message_too_fast.draw()
            
            win.flip()
                    
        elif elapsed - onset > (len_waveform[i] / config['sample_rate']): # break at the end of auditory stimulus (incl. cue, trial, and ITI)
            break
    
    # record come timing variables
    offset = PsychPortAudio('Stop', pahandle, 1) # offset measured by PsychPortAudio
    onset_sound.append(onset_soundstim - first_trigger) # onset measured by PsychPortAudio
    single_onsets = [(onset_soundstim + i * (config['duration'] + config['ISI']) - first_trigger) for i in range(config['n_tones'])] # theoretical onset of single sounds
    onset_tones.append(single_onsets) # theoretical onsets based on measured trial start
    duration_sound.append(trial_duration - config['pre_cue']) # based on getsecs

    onset_cues.append(cue_start)
    offset_cues.append(cue_end)

    #print(f'pressed: {key_name}')
    #print(f'performance: {performance[i]}')
    
    # collect key presses and RTs
    if key_pressed == True:
        key_rt_dev = key_time - single_onsets[dpos[i]] - first_trigger
        keys_pressed[i] = key_name
        rts_getsecs_trial[i] = key_rt_trial
        rts_getsecs_dev[i] = key_rt_dev
    else:
        keys_pressed[i] = None
        rts_getsecs_trial[i] = None
        rts_getsecs_dev[i] = None  

    # ----------------------------------------------------------#
    # OVERALL ACCURACY FEEDBACK TRIALS IN LAST RUN
    # ----------------------------------------------------------#
    if feedback_trial == True:
        trial_inds = np.where(np.array(runs) == runs[i])[0]
        count_correct = np.sum(np.array(performance)[trial_inds] == 'correct')
        count_slow = np.sum(np.array(performance)[trial_inds] == 'too slow') + np.sum(np.array(performance)[trial_inds] == 'correct too slow') + np.sum(np.array(performance)[trial_inds] == 'incorrect too slow')
        count_miss = np.sum(np.array(performance)[trial_inds] == 'miss')
        trials_so_far = i - (n_trials/len(np.unique(np.array(runs)))*(runs[i]-run_info+1)) + 1
        
        accuracy_now = (count_correct/trials_so_far)*100

        feedback_acc_start = ptb.GetSecs()

        while ptb.GetSecs() - feedback_acc_start <= config['feedback_duration'] + 2.5: # add some ITI after the feedback

            if ptb.GetSecs() - feedback_acc_start <= config['feedback_duration']:
                feed_acc.text = f'% korrekte Positionen in diesem Durchgang bisher: {np.abs(accuracy_now): .2f}\n\n\nZu langsam bei {count_miss + count_slow} Tonsequenzen in diesem Durchgang bisher.'
                feed_acc.draw()
            else:
                feed_acc.text = '+'
                feed_acc.draw()
            win.flip() 

# ----------------------------------------------------------#
# FINISH EXPERIMENT
# ----------------------------------------------------------#
accuracy = ((np.sum(np.array(performance) == 'correct'))/n_trials)*100

message.text = f"% korrekte Positionen in diesem Trainingsdurchgang: {np.abs(accuracy): .2f}\n\n\nBitte gib der Versuchsleitung Bescheid!"
message.color = (1, 1, 1)
message.draw()
win.flip()

now = ptb.GetSecs()

while ptb.GetSecs() < now + 3:
    core.wait(0.1)

# end experiment by pressing space
end_keys = end_kb.waitKeys(keyList=['space'], waitRelease=False)

win.close()
core.quit()