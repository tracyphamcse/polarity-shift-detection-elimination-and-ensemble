import codecs
import glob

RESOURCE_DIRECTORY = 'resources/'

def read_file(file_path):
    input = open(RESOURCE_DIRECTORY + file_path, 'r')
    file_content = input.read()
    input.close()
    return file_content

def read_file_by_utf8(file_path):
    input = codecs.open(RESOURCE_DIRECTORY + file_path, 'r', encoding='utf-8', errors='ignore')
    file_content = input.read()
    input.close()
    return file_content

def load_files_in_directory(directory):
    directory = RESOURCE_DIRECTORY + directory + '*'
    file_paths = glob.glob(directory)
    return [read_file_by_utf8(file_path.replace(RESOURCE_DIRECTORY, '')) for file_path in file_paths]

def write_file(file_path, content):
    output = open(RESOURCE_DIRECTORY + file_path, 'w')
    output.write(content)
    output.close()

def write_file_by_utf8(file_path, content):
    output = codecs.open(RESOURCE_DIRECTORY + file_path, 'w', encoding='utf-8')
    output.write(content)
    output.close()
