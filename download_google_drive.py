import gdown
import os
import patoolib

def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

def extract_rar(file_path, destination_directory):
    patoolib.extract_archive(file_path, outdir=destination_directory)

if __name__ == "__main__":
    ### Replace with the actual file ID from Google Drive
    file_id = '11P1NvO9cAM62TGgtwbPv9iUGjsx7b6IA'

    # Specify the destination directory
    destination_directory = './Dataset'
    
    os.makedirs(destination_directory, exist_ok=True)

    # # Create the destination directory if it doesn't exist
    # os.makedirs(destination_directory, exist_ok=True)

    # Specify the output filename
    output_filename = 'DataSet_from_google_drive.rar'

    # Specify the full path to the destination file
    destination_path = os.path.join(destination_directory, output_filename)

    # Download the RAR file
    download_file_from_google_drive(file_id, destination_path)
    print(f"RAR file downloaded and saved to: {destination_path}")

    # Extract the contents of the RAR file
    extract_rar(destination_path, destination_directory)

    print(f"Contents extracted to: {destination_directory}")


