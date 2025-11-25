from pathlib import Path
import numpy as np

AMASS_DIR = Path("data") / "CMU"
example_file = next(AMASS_DIR.rglob("*.npz"))
print("Example file:", example_file)

data = np.load(example_file)
print("Keys:", list(data.keys()))
