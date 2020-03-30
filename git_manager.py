import os
import os.path as path
from . import config
import subprocess


class GitManager:

    def __init__(self):

        if not path.exists(f"./{config.REPO_NAME}"):
            command = f"git clone {config.REPO_URL}"
        else:
            command = f"git pull"

        process = subprocess.Popen(command, shell=True)

        # add timeout so we do not block too much
        process_status = process.wait(10000)