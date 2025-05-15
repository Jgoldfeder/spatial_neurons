import subprocess
import os

def run_remotely(dataset,model,mode,gamma,gpu,machine,test=False):
    command = "bash finetune.sh " + str(gpu) + " " + mode + " " + str(gamma) + " " + dataset + " " + model
    if test:
        command = f"echo 'testing' >> test_{gpu}_{machine['host']}.txt && pkill python"
    ssh_cmd = [
        "ssh",
        "-i",  machine['key_path'],
        "-p", machine['port'],
        machine['host'],
        "source /venv/main/bin/activate && cd  /workspace/spatial_neurons/vision_analysis && " + command 
    ]

    result = subprocess.run(ssh_cmd, capture_output=False, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)


    remote_path = f"/workspace/spatial_neurons/vision_analysis/metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl"   # adjust if it's in a different directory
    local_path  = f"./metrics/{dataset}/{mode}/{mode}:{model}:{gamma}.pkl"  
    os.makedirs(f"./metrics/{dataset}/{mode}/", exist_ok=True)

    if test:
        remote_path = f"/workspace/spatial_neurons/vision_analysis/test_{gpu}_{machine['host']}.txt"
        os.makedirs(f"./test", exist_ok=True)
        local_path = f"./test/test_{gpu}_{machine['host']}.txt"
    # Build the scp command
    scp_cmd = [
        "scp",
        "-i", machine['key_path'],      # identity file
        "-P", machine['port'],      # port (note: scp uses uppercase -P)
        f"{machine['host']}:{remote_path}",
        local_path
    ]

    print(f"Copying {remote_path} from {machine['host']} to {local_path}…")
    # This will block until the file transfer completes (or errors)
    subprocess.run(scp_cmd, check=True)
    print("Done!")


# run_remotely(dataset,model,mode,gamma,gpu,machine)