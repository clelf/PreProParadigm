# PreProParadigm

Clémentine's part:

- Generative model class (audit_gm.py)
- Modelling RTs (model_RTs.py, compute_predictions_and_likelihoods.py, compare_RT_subj3_4sess.py)

Jasmin's part:

- Data sequence generation (generate_task_sequences(_training).py)
- Experiment implementation:
    - main task fMRI: (auditPrePro_fmri.py)
    - main task training: (auditPrePro_training.py)
    - main task refresher: (auditPrePro_refresher.py)
    - functional localizer: (auditPrePro_localizer.py)
    - sound check: (auditPrePro_sound_check.py)
    - digit span implementation (digit_span.py)
    - auxiliary things related to experimental design (ITI.py, generate_designs.py, add_itis_cues.py)