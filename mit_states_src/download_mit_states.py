#download the mit states dataset
import os
import sys
import urllib.request
import zipfile

def download_mit_states(url, root, directory):
    filename = 'StatesData.zip'
    if not os.path.exists(filename):
        print('Downloading MIT states dataset...')
        urllib.request.urlretrieve(url, filename)
    else:
        print('MIT states dataset already downloaded')
    if not os.path.exists(os.path.join(root, directory)):
        print('Extracting MIT states dataset...')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        if not os.path.exists(root):
            os.mkdir(root)
        os.system('mv release_dataset ' + root + '/' + directory)
    else:
        print('MIT states dataset already extracted')

    #move mit states to root
    

if __name__ == '__main__':
    download_mit_states()
