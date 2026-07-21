import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.generate import load_model

model = load_model()

print(model)