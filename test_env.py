from dotenv import load_dotenv
import os
import torch
import numpy as np

load_dotenv()
print("API key exists?", os.getenv("OPENAI_API_KEY") is not None)
print("Torch version:", torch.__version__)
print("NumPy version:", np.__version__)
