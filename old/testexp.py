from psychopy import prefs, sound, core

prefs.hardware['audioLib'] = ['ptb']  # fallback chain
prefs.hardware['audioDevice'] = 'Externe Kopfhörer'  # comment out temporarily
prefs.hardware['audioLatencyMode'] = 3

testSound = sound.Sound('C', secs=1.0)
testSound.play()
core.wait(1.5)