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
import random

from scipy.io.wavfile import write, read
from scipy.interpolate import PchipInterpolator

from datetime import date
import os
import sys
from glob import glob
# ----------------------------------------------------------#
# SETUP
# ----------------------------------------------------------#
config = {
    'screen_size': [1920, 1080],
    'f0': np.logspace(np.log10(600), np.log10(1300), 6),
    'freqs_loc': [200.00, 391.91, 690.41, 1154.69, 1876.82, 3000],
    'max_block': 10,
    'sample_rate': 48000,
    'isi': 0.65,
    'duration': 0.1,
    'ramp_time': 0.01,
    'k': 5,
    'a': 1
}

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

#---- open a screen and test if refresh rate can be measured
mon = monitors.Monitor('tempMonitor') 
mon.setSizePix(config['screen_size']) 

win = visual.Window(
    size=(config['screen_size'][0],config['screen_size'][1]),
    fullscr=True,
    screen=1,
    units='pix',
    color= 0,
    monitor = mon
)

escape_kb = keyboard.Keyboard(backend='ptb')

def check_escape():
    if escape_kb.getKeys(['escape'], waitRelease=False):
        try:
            PsychPortAudio('Close')
        except:
            pass
        win.close()
        core.quit()

continue_kb = keyboard.Keyboard(backend = 'ptb')    

message = visual.TextStim(win, text='+', wrapWidth=1500)
message.height = 50
message.color = (1,1,1)
message.pos = (0, 0)

message.draw()
win.flip()

interpolator = load_interpolator("weights_n_harmonics_5_a_1_duration_0.1_ramp_time_0.01_target_sone_1.csv", 'weights')

for block in range(config['max_block']):

    buffer_handles = []
    pahandle = PsychPortAudio('Open', 1, 1, 4, config['sample_rate'], 1)
    
    for i,s in enumerate(config['f0']):

        sound = generate_hct(round(s,2), config['duration'], config['sample_rate'], config['ramp_time'], config['k'], config['a'])
        sounds_localizer = glob(f"sound_check_stims/cloud_freq_{config['freqs_loc'][i]:.2f}*.wav")
        
        random.shuffle(sounds_localizer)
        sound_localizer = sounds_localizer[0]

        fs, sound_localizer = read(sound_localizer)

        weight = float(interpolator(np.log(s)))
        weighted = sound * weight

        buffer_handle = PsychPortAudio('CreateBuffer', [], weighted)
        buffer_handles.append(buffer_handle)
        
        isi = generate_isi(config['isi'], config['sample_rate'])

        buffer_handle = PsychPortAudio('CreateBuffer', [], isi)
        buffer_handles.append(buffer_handle)

        buffer_handle = PsychPortAudio('CreateBuffer', [], sound_localizer)
        buffer_handles.append(buffer_handle)

        buffer_handle = PsychPortAudio('CreateBuffer', [], isi)
        buffer_handles.append(buffer_handle)

    for i in range(len(config['f0'])*4):
        PsychPortAudio('UseSchedule', pahandle, 1)
        PsychPortAudio('AddToSchedule', pahandle, buffer_handles[i])
        onset = PsychPortAudio('Start', pahandle, 1, 0, 1)
        offset = PsychPortAudio('Stop', pahandle, 1, 0, 1)

    message.text = 'Laut genug? Auf beiden Ohren gleich laut?'    
    message.draw()
    win.flip()

    PsychPortAudio('Close')

    while True:
        check_escape()

        if continue_kb.getKeys(['space'], waitRelease=False):
            break

        core.wait(0.01)

    message.text = '+'    
    message.draw()
    win.flip()

win.close()
core.quit()