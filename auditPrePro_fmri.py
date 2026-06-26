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

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

from datetime import date, datetime
import os
import sys

devs = PsychPortAudio('GetDevices')
prefs.hardware['audioDevice'] = devs[1]
print(prefs.hardware['audioDevice'])

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
    'exp_name': 'AuditPrePro_fMRT',
    'screen_size': [1920, 1080],
    'screen_units': 'pix',
    'mouse_visible': False,
    'log_dir': 'logfiles_fmri/',
    'trial_dir': 'trial_lists/',
    'trigger_key': '5',
    'N_triggers_wait': 5,
    'sample_rate': 48000,
    'ISI': 0.65,
    'duration': 0.1,
    'ramp_time': 0.01,
    'k': 5,
    'a': 1,
    'key_pos': ['9','1','2','3','4'],
    'feedback_duration': 2,
    'response_window': 0.65,
    'correct_after_dev': 1.5,
    'pre_cue': 0.75,
    'cuesize': 50,
    'n_tones': 8,
    'n_runs': 4,
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
run_info = expInfo['run']

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
trigger_kb = keyboard.Keyboard(backend = 'ptb')
response_kb = keyboard.Keyboard(backend = 'ptb')
end_kb = keyboard.Keyboard(backend = 'ptb')

# ----------------------------------------------------------#
# SETUP FEEDBACK MESSAGE
# ----------------------------------------------------------#

feed_acc = visual.TextStim(win, text=' ', wrapWidth=1500)
feed_acc.color = (1, 1, 1)
feed_acc.height = 60
feed_acc.pos = (0, 0)

# ----------------------------------------------------------#
# SETUP CUE SHAPES
# ----------------------------------------------------------#
cuesize = config['cuesize'] 

shapes = {

    'triangle': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize, -cuesize),
            (0, cuesize),
            (cuesize, -cuesize)
        ],
        fillColor='white',
        lineColor=None
    ),
    'square': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize, -cuesize),
            (-cuesize, cuesize),
            (cuesize, cuesize),
            (cuesize, -cuesize)
        ],
        fillColor='white',
        lineColor=None
    ),
    'diamond': visual.ShapeStim(
        win,
        vertices=[
            (0, cuesize),
            (-cuesize, 0),
            (0, -cuesize),
            (cuesize, 0)
        ],
        fillColor='white',
        lineColor=None
    ),
    'circle': visual.Circle(
        win,
        radius=cuesize,
        fillColor='white',
        lineColor=None
    ),
    'hourglass': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize,  cuesize),
            ( cuesize,  cuesize),
            ( cuesize/3,  cuesize/6),
            ( cuesize/6,  0),
            ( cuesize/3, -cuesize/6),
            ( cuesize, -cuesize),
            (-cuesize, -cuesize),
            (-cuesize/3, -cuesize/6),
            (-cuesize/6,  0),
            (-cuesize/3,  cuesize/6),
        ],
        fillColor='white',
        lineColor=None,
        closeShape=True
    ),
    'dome': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize, -cuesize),  # bottom left
            (cuesize, -cuesize),   # bottom right
            (cuesize, 0),          # right side up to diameter
        ] + [
            (cuesize * np.cos(t), cuesize * np.sin(t))
            for t in np.linspace(0, np.pi, 100)
        ] + [
            (-cuesize, 0),         # left side of diameter
        ],
        fillColor='white',
        lineColor=None,
        interpolate=True
    ),
    'x': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize, -cuesize/2),
            (-cuesize/2, -cuesize),
            (0, -cuesize/2),
            (cuesize/2, -cuesize),
            (cuesize, -cuesize/2),
            (cuesize/2, 0),
            (cuesize, cuesize/2),
            (cuesize/2, cuesize),
            (0, cuesize/2),
            (-cuesize/2, cuesize),
            (-cuesize, cuesize/2),
            (-cuesize/2, 0)
        ],
        fillColor='white',
        lineColor=None
    ),
    'star': visual.ShapeStim(
        win,
        vertices=[
            (0, cuesize),
            (-cuesize/4, cuesize/4),
            (-cuesize, 0),
            (-cuesize/4, -cuesize/4),
            (0, -cuesize),
            (cuesize/4, -cuesize/4),
            (cuesize, 0),
            (cuesize/4, cuesize/4)
        ],
        fillColor='white',
        lineColor=None
    ),
    'crown': visual.ShapeStim(
        win,
        vertices=[
            (-cuesize, -cuesize),
            (-cuesize, cuesize/2),
            (-cuesize/2, 0),
            (0, cuesize/2),
            (cuesize/2, 0),
            (cuesize, cuesize/2),
            (cuesize, -cuesize)
        ],
        fillColor='white',
        lineColor=None
    ),
    'heart': visual.ShapeStim(
        win,
        vertices=[
            (0, -cuesize),

            (-0.6*cuesize, -0.3*cuesize),
            (-0.9*cuesize, 0.2*cuesize),
            (-0.8*cuesize, 0.6*cuesize),
            (-0.4*cuesize, 0.9*cuesize),
            (0, 0.6*cuesize),

            (0.4*cuesize, 0.9*cuesize),
            (0.8*cuesize, 0.6*cuesize),
            (0.9*cuesize, 0.2*cuesize),
            (0.6*cuesize, -0.3*cuesize),

            (0, -cuesize)
        ],
        fillColor='white',
        lineColor=None,
        closeShape=True
    ),
    'hexagon': visual.Polygon(
        win,
        edges=6,
        radius=cuesize,
        fillColor='white',
        lineColor=None
    ),
    'flower': visual.ShapeStim(
        win,
        vertices=[
            (0, cuesize),

            (-cuesize * 0.25, cuesize * 0.55),
            (-cuesize * 0.7, cuesize * 0.7),
            (-cuesize * 0.55, cuesize * 0.25),

            (-cuesize, 0),

            (-cuesize * 0.55, -cuesize * 0.25),
            (-cuesize * 0.7, -cuesize * 0.7),
            (-cuesize * 0.25, -cuesize * 0.55),

            (0, -cuesize),

            (cuesize * 0.25, -cuesize * 0.55),
            (cuesize * 0.7, -cuesize * 0.7),
            (cuesize * 0.55, -cuesize * 0.25),

            (cuesize, 0),

            (cuesize * 0.55, cuesize * 0.25),
            (cuesize * 0.7, cuesize * 0.7),
            (cuesize * 0.25, cuesize * 0.55),
        ],
        fillColor='white',
        lineColor=None,
        closeShape=True
    )
}

# ----------------------------------------------------------#
# INITIALIZE VARS
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Starte Experiment ...', wrapWidth=1500)
message.height = 60
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
trials = pd.read_csv(f"{config['trial_dir']}sub-{participant}/sub-{participant}_ses-{session}_run-{run_info}_trials.csv")

n_trials = len(trials)/config['n_tones']
rts_getsecs_trial = [None]*(int(n_trials)) # RT based on ptb.getSecs() from onset trial
rts_getsecs_dev = [None]*(int(n_trials)) # RT based on ptb.getSecs() from onset deviant
keys_pressed = [None]*(int(n_trials)) # pressed key
performance = [None]*(int(n_trials)) # correct?

#---- more important variables
len_waveform = [] # len waveform used to compute duration of audio playback
buffer_handles = [] # buffer for audido playback

# create all sound stimuli (incl. max amplitude weighing and loudness normalization)
weight_list = pd.read_csv("weights_n_harmonics_5_a_1_duration_0.1_ramp_time_0.01_target_sone_1.csv")

#sound_raw = [[] for _ in range(len(pd.unique(trials["trial_n"])))]
sound_norm = [[] for _ in range(len(pd.unique(trials["trial_n"])))]
f0s_all = [[] for _ in range(len(pd.unique(trials["trial_n"])))]

for i in pd.unique(trials["trial_n"]):
    freq_data = trials[trials["trial_n"] == i]
    for s in freq_data["frequency"]:

        f0s_all[i].append(s)

        raw = generate_hct(
            round(s, 2),
            config['duration'],
            config['sample_rate'],
            config['ramp_time'],
            config['k'],
            config['a']
        )

        #sound_raw[i].append(raw)

        #norm = raw / weight_list['max_amp'][0]
        sound_norm[i].append(raw)


# loudness equalization!!! only works if the CSV file containing the weight was created with the exact tone duration and harmonic configuration as in config here!!!
sound_loud_weight = [[] for _ in range(len(pd.unique(trials["trial_n"])))] 
interpolator = load_interpolator("weights_n_harmonics_5_a_1_duration_0.1_ramp_time_0.01_target_sone_1.csv", 'weights')

for i in pd.unique(trials["trial_n"]):
    freq_data = trials[trials["trial_n"] == i]
    for n,s in enumerate(freq_data["frequency"]):
        freq = f0s_all[i][n]
        print(freq)
        weight = float(interpolator(np.log(freq)))
        print(weight)
        weighted = sound_norm[i][n] * weight
        sound_loud_weight[i].append(weighted)

#---- open audio port
pahandle = PsychPortAudio('Open', 1, 1, 4, config['sample_rate'], 1) # TODO: watch out, might need to adjust to final machine!

#---- load experiment: create all trials in advance from previously generated trial_list
for i in pd.unique(trials["trial_n"]):
    
    message = visual.TextStim(win, text='Lade Stimuli ...', wrapWidth=1500)
    message.height = 60
    message.draw()
    win.flip()

    temp = trials[trials["trial_n"] == i]
    
    waveform = []
    oddball_trial = temp['frequency']
    
    dpos_trial = temp['dpos']
    dposy = dpos_trial.iloc[0]
    
    if dposy != 0:
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
    if dposy != 0:
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
        waveform.append(sound_loud_weight[i][n])
        
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

# ----------------------------------------------------------#
# WAIT FOR TRIGGER
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Warten auf den Scanner...', wrapWidth=1500)
message.height = 60
message.draw()
win.flip()

trigger_list = [config['trigger_key']]
n_triggers_to_wait = config['N_triggers_wait'] # how many kay presses to wait for
trigger_times = [] # trigger_times[0] would be the time to subtract from all onsets, save all 5 in case we need to get rid of non-steady-state images

for t in range(n_triggers_to_wait): # start on the first space press for behavioral experiment
    trigger_keys = trigger_kb.waitKeys(keyList=trigger_list, waitRelease=False) # TODO: adjust for scanning
    trigger_time = ptb.GetSecs()
    trigger_times.append(trigger_time)

trigger_times_zero = [x - trigger_times[0] for x in trigger_times]    
first_trigger = trigger_times[0] # this value is used to compute all onsets    

timer_start = core.CountdownTimer(4.0)

while True:
    time_left = timer_start.getTime()

    if time_left <= 0.:
        break
    else:
        secs = int(time_left % 60)
        if secs != 0:
            message.text = f'{secs:01}'
            message.height = 60
            message.pos = (0, 0)
            message.draw()
            win.flip()
        if secs == 0:    
            message.text = '+'
            message.height = 60
            message.pos = (0, 0)
            message.draw()
            win.flip()

# ----------------------------------------------------------#
# RUN EXPERIMENT
# ----------------------------------------------------------#
feedback_trial = False

for i in range(0, int(n_trials) + 1):

    # add indicator for feedback trials (every 6th trial but not at end of run)
    if i == 0 or i == n_trials:
        feedback_trial = False
        feedback_trialy.append(0)
    if (i+1) % 6 == 0 and i > 0 and i < (n_trials) - 1:
        feedback_trial = True
        feedback_trialy.append(1)
    elif (i+1) % 6 != 0 and i > 0 and i < (n_trials) - 1:
        feedback_trial = False
        feedback_trialy.append(0)
    
    if i == (n_trials):
        
        # create logfile
        data = []

        print(len(onset_sound), len(onset_tones), len(duration_sound), 
              len(ITI_list), len(rts_getsecs_trial), len(rts_getsecs_dev), len(tau), len(frequency), len(lim_std), len(lim_dev),len(keys_pressed),
              len(performance), len(rule),len(dpos),len(cue),len(tone_type),len(runs),len(trial_nr),len(feedback_trialy)) 
        
        # write logfile
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
            'feedback_trial': np.repeat(feedback_trialy,config['n_tones']),
            'ITI': trials['ITI'],
            'tau_std': np.repeat(tau,config['n_tones']),
            'lim_std': np.repeat(lim_std,config['n_tones']),
            'lim_dev': np.repeat(lim_dev,config['n_tones']), 
            'trial': np.repeat(trial_nr,config['n_tones']),
            'run': np.repeat(runs,config['n_tones']),
            'onset_cues':  np.repeat(onset_cues,config['n_tones']),
            'offset_cues':  np.repeat(offset_cues,config['n_tones']),
            'trigger_times': [trigger_times] * int(len(frequency)),
            'trigger_times_zero': [trigger_times_zero] * int(len(frequency)),
        })

        data.to_csv(f"{config['log_dir']}/sub-{participant}_ses-{session}_task-AuditPrePro_run-{run_info}_events-{date}.tsv", sep='\t', index=False)
        break
        
    #---- prepare sound playback
    phase ='stimulus'
    key_pressed = False
    feedback_recorded = False
    feedback_start = None
    response_start = None
    key_name = None
    key_rt_trial = None
    key_rt_dev = None
    cue_start = None
    cue_end = None

    # play audio sequences from buffer
    PsychPortAudio('UseSchedule', pahandle, 1)  # 1 = replace current schedule
    PsychPortAudio('AddToSchedule', pahandle, buffer_handles[i])
    onset = PsychPortAudio('Start', pahandle, 1, 0, 1) # measure onset trial based on PsychPortAudio
    cue_start = onset - first_trigger
    onset_soundstim = onset + config['pre_cue']

    time_to_dev = (dpos[i]*config['duration']) + (dpos[i]*config['ISI'])
    time_to_dev_all = [((np.unique(dpos)[j-2]*config['duration']) + (np.unique(dpos)[j-2]*config['ISI'])) for j in np.unique(dpos)]
    time_to_dev_3 = (2*config['duration']) + (2*config['ISI'])
    
    response_kb.clearEvents()  # clear any leftover keys

    show_shape = True

    # ----------------------------------------------------------#
    # IMPLEMENT REGULAR TRIAL
    # ----------------------------------------------------------#
    while True and dpos[i] != 0:
        
        elapsed = ptb.GetSecs() 

        # record key presses continuously
        if key_pressed == False: # only record first response

            response_keys = response_kb.getKeys(config['key_pos'], waitRelease=False)

            if response_keys and not key_pressed:
                key_pressed = True
                key_name = response_keys[0].name
                key_time = ptb.GetSecs()
                key_rt_trial = key_time - onset_soundstim # rt relative to trial onset --> compute relative to deviant onset later

            if not feedback_recorded:
                if dpos[i] != 0 and key_pressed:
                    correct_key = dpos[i] - 2 # get correct answer
                    key_posy = config['key_pos'].index(key_name)
                    
                    if correct_key == key_posy and key_rt_trial > time_to_dev: # in case of correct answer
                        
                        if (key_rt_trial - time_to_dev_all[key_posy]) <= config['correct_after_dev']:
                            performance[i] = 'correct'
                            feedback_recorded = True
                            
                        elif (key_rt_trial - time_to_dev_all[key_posy]) > config['correct_after_dev']: # if RT > 1.5 seconds after deviant onset --> too slow
                            performance[i] = 'correct too slow'
                            feedback_recorded = True 
                        
                    elif correct_key != key_posy and key_rt_trial > time_to_dev_3: # in case of incorrect answer happening after position 3

                        if key_rt_trial > time_to_dev_all[key_posy]: # if incorrectly responded after deviant

                            if (key_rt_trial - time_to_dev_all[key_posy]) <= config['correct_after_dev']:
                                performance[i] = 'incorrect' 
                                feedback_recorded = True

                            elif (key_rt_trial - time_to_dev_all[key_posy]) > config['correct_after_dev']: # or too slow feedback if after RT window
                                performance[i] = 'incorrect too slow'
                                feedback_recorded = True 

                        elif key_rt_trial <= time_to_dev_all[key_posy]:
                            performance[i] = 'too fast'
                            feedback_recorded = True

                    elif correct_key == key_posy and key_rt_trial < time_to_dev: # too fast if correct answer is given before the dev actually occurs
                        performance[i] = 'too fast'
                        feedback_recorded = True

                    elif key_rt_trial < time_to_dev_3: # too fast if response before tone 3
                        performance[i] = 'too fast'
                        feedback_recorded = True
                        
        
        # make experiment closable by Esc key press
        if 'escape' in [k.name for k in escape_kb.getKeys(['escape'], waitRelease=False)]:
            core.quit(); sys.exit()
        
        # ----------------------------------------------------------#
        # PRESENT CUE
        # ----------------------------------------------------------#
        elif elapsed - onset <= (len_waveform[i] / config['sample_rate']):

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
                    onset_iti = ptb.GetSecs()
                  
            # ----------------------------------------------------------#
            # SMALL RESPONSE WINDOW AFTER LAST TONE (CUE STILL PRESENTED)
            # ----------------------------------------------------------#
            elif phase == 'response':

                # small buffer to avoid visual overlap
                if show_shape and (elapsed-onset) > (trial_duration + config['response_window'] - 0.05):
                    show_shape = False
                    cue_end = ptb.GetSecs() - first_trigger
                              
                # collect feedback (text not shown in main experiment)
                if (elapsed-onset) > (trial_duration + config['response_window']):

                    if key_pressed:
                        phase = 'ITI'
                    elif not key_pressed:
                        performance[i] = 'miss' # missed response = 4 (attention: previously correct omission)
                        feedback_recorded = True
                        phase = 'ITI'

            # ----------------------------------------------------------#
            # ITI
            # ----------------------------------------------------------#
            elif phase == 'ITI':
                message.text = '+'
                message.draw()                           

            win.flip()
                    
        elif elapsed - onset > (len_waveform[i] / config['sample_rate']):
            offset_iti = ptb.GetSecs()
            break
    
    offset = PsychPortAudio('Stop', pahandle, 1) # offset measured by PsychPortAudio
    onset_sound.append(onset_soundstim-first_trigger)
    single_onsets = [(onset_soundstim + i * (config['duration'] + config['ISI'])-first_trigger) for i in range(config['n_tones'])]
    onset_tones.append(single_onsets) # theoretical onsets based on measured trial start
    duration_sound.append(trial_duration - config['pre_cue']) # theoretical trial duration (excluding pre_cue)

    onset_cues.append(cue_start)
    offset_cues.append(cue_end)

    print(f'pressed: {key_name}')
    print(f'performance: {performance[i]}')
    
    if dpos[i] != 0:
        pass
    else:
        performance[i] = None
    
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
    # OVERALL ACCURACY FEEDBACK TRIALS (every 6th trial)
    # ----------------------------------------------------------#
    if feedback_trial == True:

        trial_inds = np.where(
            (np.array(runs) == runs[i]) &
            (np.array(dpos) != 0)
        )[0]

        count_correct = np.sum(np.array(performance)[trial_inds] == 'correct')
        count_slow = np.sum(np.array(performance)[trial_inds] == 'too slow') + np.sum(np.array(performance)[trial_inds] == 'correct too slow') + np.sum(np.array(performance)[trial_inds] == 'incorrect too slow')
        count_miss = np.sum(np.array(performance)[trial_inds] == 'miss')
        trials_so_far = i + 1
        
        accuracy_now = (count_correct/trials_so_far)*100

        feedback_start = ptb.GetSecs()

        while ptb.GetSecs() - feedback_start <= config['feedback_duration'] + 2.5:

            if ptb.GetSecs() - feedback_start <= config['feedback_duration']:
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

timer_end = core.CountdownTimer(15.0)

while True:
    time_left = timer_end.getTime()

    if time_left <= 0:
        break
    else:
        secs = int(time_left % 60)
        message.text = f"% korrekte Positionen insgesamt: {np.abs(accuracy): .2f}\n\n\nEnde! Bitte noch {secs:02} Sekunden kurz warten!"
        message.draw()
        win.flip()

full_duration = ptb.GetSecs()-first_trigger
print(f"full scanning duration: {full_duration/60} minutes")        

# end experiment by pressing space
end_keys = end_kb.waitKeys(keyList=['space'], waitRelease=False)

win.close()
core.quit()