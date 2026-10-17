__version__ = "0.1.0"

from posttrain_lab.utils.dotenv import export_hf_env

export_hf_env()  # HF_HOME / HF_ENDPOINT / HF_*_OFFLINE from .env, before any HF import
