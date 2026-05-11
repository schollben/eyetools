# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".utils"))
from load_session_data import load_session_data

SESSION = "session_2025-07-07_ferret_753_EyeCameras_P39_E11_analyzable_output"

Results = load_session_data(SESSION)

# %%

