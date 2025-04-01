import os
import time


def delete_temp_files(temp_dir: str):
    """ Safe deletion of temporary files.
    """
    max_retries = 2
    for i in range(max_retries):
        try:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(temp_dir)
            break
        except PermissionError:
            time.sleep(0.2 * (i+1)) # waiting to close all files
        except FileNotFoundError:
            break
