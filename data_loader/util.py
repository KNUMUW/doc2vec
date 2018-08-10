import os


def get_files_from_dir(dir_path):
    """Returns all data file paths from given directory.

    Args:
        dir_path (str): the directory we want files from. 
    Returns:
        file_paths (list(str)): paths to data files in given directory.   

    """ 
    
    file_paths = []

    for entry in os.scandir(dir_path):
        if entry.is_file():
            file_paths.append(entry.path)

    return file_paths

