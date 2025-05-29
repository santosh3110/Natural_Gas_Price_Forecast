from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
EIA_API_KEY = os.getenv("EIA_API_KEY")