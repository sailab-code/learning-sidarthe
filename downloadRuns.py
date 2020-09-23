import os
import subprocess
import os.path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser("Download runs from a remote server")

parser.add_argument('--remote', metavar="remote_server", type=str, dest="remote", help="The address of the remote server in format username@url:port")
parser.add_argument('--remote_root', metavar="remote_root", type=str, dest="remote_root", help="The root of the learning sidarthe suite in the remote server", default="covid-tools")
parser.add_argument('--runs_directory', metavar="runs_directory", type=str, dest="runs_directory", help="The runs directory to download")
parser.add_argument('--model', metavar="model", type=str, dest="model", help="The model name", default="sidarthe")
parser.add_argument('--region', metavar="region", type=str, dest="region", help="The region of the runs", default="Italy")


if __name__ == "__main__":
    args = parser.parse_args()
    #ls_runs_command = ["ssh", "-t", args.remote, f"ls -1 {args.remote_root}/runs/{args.runs_directory}/{args.model}/{args.region}"]
    #ls_process = subprocess.run(ls_runs_command, capture_output=True, shell=False, text=True)
    
    run_dir_path = f"{args.runs_directory}/{args.model}/{args.region}/"
    local_run_path = f"{run_dir_path}"

    if not os.path.exists(local_run_path):
        os.makedirs(local_run_path)

    remote_run_path = f"{args.remote}:{args.remote_root}/{run_dir_path}"

    rsync_cmd = ["rsync", "-zhae", "ssh", "--exclude", "*tfevents*", f"{remote_run_path}", f"{local_run_path}"]
    print("Executing command \"" + " ".join(rsync_cmd) + "\"")

    subprocess.run(rsync_cmd, shell=True)

       
