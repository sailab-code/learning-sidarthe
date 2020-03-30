import os
import os.path as path
import subprocess

REPO_URL="https://github.com/pcm-dpc/COVID-19.git"
REPO_NAME="COVID-19"

class GitManager:
    def __init__(self):

        if not path.exists(f"./{REPO_NAME}"):
            command = f"git clone {REPO_URL}"
        else:
            command = f"git pull"

        process = subprocess.Popen(command, shell=True)

        # add timeout so we do not block too much
        process_status = process.wait(10000)