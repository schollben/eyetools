# %% LOAD DATA SCRIPT
# builds and populates the Results object with data from normal and delayed IDs
# use for all analyses
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import os
import pickle
from analyses.helper_functions import load_results

# session IDs for normal experience recordings
# issues? 
# session_2025-07-09_ferret_753_EyeCameras_P41_E13
NORMAL_IDS = (402, 405, 407, 420, 753, 757)

# delayed experience IDS (note we are targeting early Experience recordings)
DELAYED_IDS = (416, 403, 411)

Results = load_results(NORMAL_IDS + DELAYED_IDS)
fileSaveName = os.path.join(local_config.EYETOOLS_ROOT, "analyses", "results.pkl")
with open(fileSaveName, "wb") as f:
    pickle.dump(Results, f)    
