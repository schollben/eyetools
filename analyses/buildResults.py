# %% LOAD DATA SCRIPT
# builds and populates the Results object with data from normal and delayed IDs
# use for all analyses
from analyses.helper_functions import load_results

NORMAL_IDS = (402, 405, 407, 420, 753, 757)

# delayed experience IDS (note we are targeting early Experience recordings): 
# session_2026-03-10_ferret_411_P43_E2_analyzable_output
# session_2026-03-10_ferret_403_P43_E1_analyzable_output
# session_2026-03-07_ferret_416_P40_E1_analyzable_output
DELAYED_IDS = (416, 403, 411)

Results = load_results(NORMAL_IDS + DELAYED_IDS)
