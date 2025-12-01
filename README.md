# PreProParadigm

Clémentine's part:

- Generative model class

Jasmin's part:

- Data sequence generation (generate_task_sequences.py)
- Experiment (auditPrePro_exp_behavioral.py)


USAGE NOTES EXPERIMENT (auditPrePro_exp_behavioral.py)

- run generate_task_sequences.py with desired participant number to generate task sequences

- in experiment script change prefs.hardware['audioDevice'] to your local audio device
- if necessary change sample_rate to sampling rate of your device
- if necessary change your local keyboard layout to US
- if necessary change screen window size

- task: listen to sequences of eight sounds (possible deviant in any position 3-7), rarely, trials don't have a deviant
- as soon as asked to answer, identify the position of the deviant via keypress
- keys: v,z,u,i,l (German layout) or v,y,u,i,l (US layout)
- use the right hand to answer: v = position 3, z/y = position 4, etc.
- response window = 1.5 seconds
- after response or no response: indicate how confident you are in your choice using the same keys
- at the beginning and after each block: press space to continue