# ----------------------------------------------------------#
# CHECK PSYCHOPY VERSION
# ----------------------------------------------------------#
import psychopy
print(f"Running PsychoPy {psychopy.__version__}")

from psychopy import prefs
prefs.hardware['keyboard'] = 'ptb'
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'

# ----------------------------------------------------------#
# IMPORTS
# ----------------------------------------------------------#
from psychopy import visual, sound, event, core, monitors, gui
from psychopy.hardware import keyboard

from collections import defaultdict
import pandas as pd
import numpy as np
import os

# ----------------------------------------------------------#
# CONFIGURATIONS; SETUP SPEAKERS, GUI, AND LOGFILE DIR
# ----------------------------------------------------------#
config = {'screen_size': [3840, 2160], # change to your screen size
          'wrap_width': 3000, # should be smaller than screen_size[0]
          'font_height_instructions': 60, # adjust in case instructions are cut of due to smaller screen sizes
          'exp_name': 'digit_span',
          'log_dir': 'logfiles_digit_span/',
          'speakers': 'Externe Kopfhörer'} # change to your headphones

prefs.hardware['audioDevice'] = config['speakers']

os.makedirs(f"{config['log_dir']}", exist_ok=True)

expName =  config['exp_name']
expInfo = {'participant':''}

dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,title=expName)
if dlg.OK == False:
    core.quit()

participant = expInfo['participant']

# ----------------------------------------------------------#
# SETUP MONITOR
# ----------------------------------------------------------#
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

refresh_rate = win.getActualFrameRate(
    nIdentical=20, nMaxFrames=200, nWarmUpFrames=10, threshold=1
)

if refresh_rate:
    print(f"Refresh rate: {refresh_rate:.2f} Hz")
else:
    print("Could not measure refresh rate.")

# ----------------------------------------------------------#
# SETUP FILES
# ----------------------------------------------------------#
audioFiles = [
    'digit_span_stims/DS_intonM_dig_sequ3a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ3b.mp3',
    'digit_span_stims/DS_intonM_dig_sequ4a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ4b.mp3',
    'digit_span_stims/DS_intonM_dig_sequ5a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ5b.mp3',
    'digit_span_stims/DS_intonM_dig_sequ6a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ6b.mp3',
    'digit_span_stims/DS_intonM_dig_sequ7a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ7b.mp3',
    'digit_span_stims/DS_intonM_dig_sequ8a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ8b.mp3',
    'digit_span_stims/DS_intonM_dig_sequ9a.mp3',
    'digit_span_stims/DS_intonM_dig_sequ9b.mp3'
]

# ----------------------------------------------------------#
# CORRECT ANSWERS
# ----------------------------------------------------------#
seq_len = [3,3,4,4,5,5,6,6,7,7,8,8,9,9]
seqs = ['582','694','7286','8539','42731','75836','619473','392487','5917428','4179386','58192647','38295174','268415793','642718539']
correct = []

# ----------------------------------------------------------#
# INSTRUCTIONS
# ----------------------------------------------------------#
start_kb = keyboard.Keyboard(backend = 'ptb')
escape_kb = keyboard.Keyboard(backend='ptb')
end_kb = keyboard.Keyboard(backend='ptb')

message = visual.TextStim(win, text='In dieser Aufgabe werden Ihnen Sequenzen von  Zahlen vorgespielt, ' \
'die mit ein paar Zahlen beginnen und sich dann allmählich in ihrer Länge steigern.\n\n Jede Sequenz enthält Zahlen zwischen 1 und 9. ' \
'Direkt nach dem Ende jeder Sequenz erscheint ein Textfeld und Sie sollen alle Zahlen der gehörten Sequenz genau in der Reihenfolge eingeben, '
'in der Sie sie gehört haben. Vor jeder Zahlensequenz ertönt ein Signalton.\n\nBitte beachten Sie, dass Sie sich jede Sequenz nur einmal anhören und' \
' erst dann mit der Eingabe der Zahlen beginnen können, wenn die gesamte Zahlensequenz abgespielt wurde. ' \
'Lassen Sie bei Ihrer Eingabe keine Zahl aus, auch wenn Sie sich unsicher sind. Bestätigen Sie Ihre Eingabe durch "Weiter".' \
' Der "Weiter"-Button wird erst erscheinen, wenn Sie die korrekte Anzahl an Zahlen eingegeben haben.\n\n\n\nDrücken Sie die Leertaste, um zu starten!', wrapWidth = config['wrap_width'])

message.height = config['font_height_instructions']
message.pos = (0, 0)
message.draw()
win.flip()

start_keys = start_kb.waitKeys(keyList=['space'], waitRelease=False)

message.text = ' '
message.height = 100
message.draw()
win.flip()

# ----------------------------------------------------------#
# PLAY SOUNDS AND RECORD ANSWERS
# ----------------------------------------------------------#
all_responses = []

for N, audioFile in enumerate(audioFiles):

    snd = sound.Sound(audioFile, stereo = True)

    start_mouse = event.Mouse()

    startIcon = visual.ImageStim(
            win,
            image='digit_span_stims/play.png',
            pos=(0, 0),
            size=(600, 300)
        )
    
    while True:
        startIcon.draw()
        win.flip()

        if start_mouse.isPressedIn(startIcon):
            win.flip()
            break

    snd.play()
    core.wait(snd.getDuration())

    textbox = visual.TextBox2(
        win,
        text='',
        editable=True,
        pos=(0, 0),
        size=(600, 200),
        letterHeight=50,
        fillColor='white',
        color = 'black',
        borderColor='black',
        borderWidth=2,
        placeholder = 'Bitte Zahlen eingeben'
    )

    continueIcon = visual.ImageStim(
        win,
        image='digit_span_stims/weiter.png',
        pos=(150, -250),
        size=(300, 150)
    )

    continue_mouse = event.Mouse()

    while True:

        if 'escape' in [k.name for k in escape_kb.getKeys(['escape'], waitRelease=False)]:
            core.quit(); sys.exit()

        textbox.draw()
        textbox.hasFocus = True

        if len(textbox.text) == seq_len[N]:
            continueIcon.draw()

        win.flip()

        if continue_mouse.isPressedIn(continueIcon):
            all_responses.append(textbox.text)
            textbox.text = ''
            win.flip()
            break

    if all_responses[N] == seqs[N]:
        correct.append(1)
    else:    
        correct.append(0)

# ----------------------------------------------------------#
# COMPUTE DIGIT SPAN
# ----------------------------------------------------------#
grouped = defaultdict(list)

for s, a in zip(seq_len, correct):
    grouped[s].append(a)

all_correct_levels = [] # span for two correct answers
one_correct_levels = [] # span for one correct answer

for span, acc in grouped.items():
    if all(a == 1 for a in acc):
        all_correct_levels.append(span)
        one_correct_levels.append(span)
    elif sum(acc) == 1:
        one_correct_levels.append(span)
    elif sum(acc) == 0:
        one_correct_levels.append(0)    

span_all_correct = max(all_correct_levels, default=None)
span_one_correct = max(one_correct_levels, default=None)

# ----------------------------------------------------------#
# SAVE DATA
# ----------------------------------------------------------#
df = pd.DataFrame({
    'answer': all_responses,
    'accuracy': correct,
    'span_one': np.repeat(span_one_correct, len(correct)),
    'span_all': np.repeat(span_all_correct, len(correct))
})

df.to_csv(f"{config['log_dir']}/sub-{participant}_task-digitspan.tsv", sep='\t', index=False)

# ----------------------------------------------------------#
# END EXPERIMENT
# ----------------------------------------------------------#
message = visual.TextStim(win, text='Ende! Bitte geben Sie der Versuchsleitung Bescheid.', wrapWidth=2000)
message.height = 60
message.draw()
win.flip()

end_keys = end_kb.waitKeys(keyList=['space'], waitRelease=False) # end with space press

snd.stop()
win.close()
core.quit()