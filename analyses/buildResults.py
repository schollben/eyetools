# %% LOAD DATA SCRIPT
# builds and populates the Results object with data from normal and delayed IDs
# use for all analyses

from analyses.helper_functions import load_results
NORMAL_IDS = (402, 405, 407, 420, 753, 757) 
DELAYED_IDS = (416, 403, 411)
Results = load_results(NORMAL_IDS + DELAYED_IDS)
