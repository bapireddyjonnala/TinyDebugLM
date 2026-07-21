import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.generate import TinyDebugGenerator

generator = TinyDebugGenerator()

token = generator.predict_next_token(
    "ModuleNotFoundError: No module named torch"
)

print("Predicted Token ID:", token)