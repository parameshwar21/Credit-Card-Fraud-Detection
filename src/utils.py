import os

def ensure_directories():
    dirs = ["models", "results"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
