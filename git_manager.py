import os
import os.path as path
import config
import subprocess


class GitManager:

    def __init__(self):

        if not path.exists(f"./{config.REPO_NAME}"):
            process = subprocess.Popen(f"git clone {config.REPO_URL}")
        else:
            process = subprocess.Popen(f"git pull")

        # add timeout so we do not block too much
        process_status = process.wait(10000)