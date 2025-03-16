import subprocess

def git_clone(repository_url, destination_path):
    try:
        # Run the git clone command
        subprocess.run(['git', 'clone', repository_url, destination_path], check=True)
        print(f"Repository cloned successfully to {destination_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

# Change accordingly
repository_url = 'https://github.com/facebookresearch/Mask2Former.git'
destination_path = './Mask2Former' # important to set a new directory to which all files will be transfered

git_clone(repository_url, destination_path)
