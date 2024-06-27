import os
from dotenv import load_dotenv

load_dotenv()

model_path = os.getenv("MODEL_PATH")
GPU = os.getenv("GPU")
