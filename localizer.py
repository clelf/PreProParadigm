from psychopy import prefs
prefs.general['version'] = '2025.1.1'
prefs.hardware['keyboard'] = 'ptb'
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4' # could also use 4 but then no fallback in case of small deviation
#prefs.hardware['audioDevice'] = 'Externe Kopfhörer' # cable headphones
prefs.hardware['audioDevice'] = 'Mac mini-Lautsprecher' # mac mini speakers
#prefs.hardware['audioDevice'] = 'Speakers (Realtek HD Audio output)'

#---- check psychopy version
from psychopy import useVersion
#useVersion('2025.1.1')

import psychopy
print(f"Running PsychoPy {psychopy.__version__}")

from psychopy import sound
from psychtoolbox import PsychPortAudio

#---- imports
import psychtoolbox as ptb
from psychtoolbox import audio
from psychtoolbox import PsychPortAudio
PsychPortAudio('Close')

from psychopy import sound
import sounddevice as sd
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

os.makedirs('logfiles_localizer/', exist_ok=True)

#---- open GUI to enter participant information
expName = 'AuditPrePro'
expInfo = {'participant':'', 'session':''} # enter participant and session info as defined in generate_task_sequences.py


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
mon.setSizePix([2569, 1440]) # set to size of actual monitor

win = visual.Window(
    size=(2569, 1440),
    fullscr=False, # True for actual running
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
    
message = visual.TextStim(win, text='Lade stimuli ...', wrapWidth=2000)
message.height = 100
message.draw()
win.flip()

