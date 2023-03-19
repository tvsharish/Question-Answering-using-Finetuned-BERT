import urllib.request
import zipfile,sys,os

def download_zip():
    file_path=os.path.join(sys.path[0],'tqa_v1_train.json')
    if not os.path.exists(file_path):
        url = 'https://ai2-public-datasets.s3.amazonaws.com/tqa/tqa_train_val_test.zip'
        filehandle, _ = urllib.request.urlretrieve(url)
        zip_object = zipfile.ZipFile(filehandle, 'r')
        first_file = zip_object.open["tqa_train_val_test/train/tqa_v1_train.json"]
        with open(file_path,'w') as f:
            f.write(str(first_file.read()))
        return first_file.read()
    else:
        return open(file_path,'r')    