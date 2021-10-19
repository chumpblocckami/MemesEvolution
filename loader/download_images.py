import pandas as pd 
import os 
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm 
import numpy as np
import time 
import logging 
import multiprocessing as mp
logging.basicConfig(filename='./../memes_download.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

class Loader():
    def __init__(self, path_2_file,path_2_dataset):
        self.logger = logging.getLogger('Download')

        self.path_2_file = path_2_file
        self.path_2_dataset = path_2_dataset
        self.data = pd.DataFrame()

    def fit(self,):
        for csv in tqdm(os.listdir(path_2_file),desc='Loading images urls'):
            if '.csv' in csv:
                _data = pd.read_csv(path_2_file+'/'+csv, header=None, low_memory=False)
                if _data.columns.dtype == int:
                    _data.columns = _data.iloc[0]
                    _data = _data.iloc[1:]
                self.data = self.data.append(_data)

    def download_batch(self, process, lower_bound, upper_bound):
        for _, row in tqdm(self.data.iloc[lower_bound:upper_bound].iterrows(),desc='Process n'+str(process)+' is downloading...'):
            if str(row['id'])+'.png' in os.listdir(self.path_2_dataset):
                continue
            try:
                response = requests.get(row['url'])

            except Exception as _ConnectionError:
                self.logger.error('Error getting ' + str(row['id']) + ' image. Full log:'+str(_ConnectionError))
                continue 
            try:
                img = Image.open(BytesIO(response.content))
                img.save(self.path_2_dataset+'/'+str(row['id'])+'.png')
                self.logger.info('Process '+ str(process) +' has saved images with id ' + str(row['id']))

            except IOError:
                try:
                    img.convert('RGB').save(self.path_2_dataset+'/'+str(row['id'])+'.png',optimize=True)
                except Exception as GenericError:
                    self.logger.error('Error saving the ' + str(row['id']) + ' image. Full log:'+str(GenericError))
                    continue
            time.sleep(np.random.uniform(0,1)) 

    def run_multiprocess_download(self,n_instances):
        n_memes = self.data.shape[0]
        batches_size = range(0,n_memes,int(n_memes/n_instances))
        self.logger.info('Downloading '+str(n_memes)+' memes using '+str(n_instances)+' istances.')
        excess = (n_memes - max(batches_size))
        processes = []
        for pid in range(n_instances):
            lower = batches_size[pid]
            upper = batches_size[pid+1]-1 + excess if pid == n_instances else batches_size[pid+1]-1
            processes.append(
                mp.Process(target = self.download_batch,
                args=(pid, lower, upper))
                )

        for p in processes:
            p.start()
            time.sleep(1)
            
        for p in processes:
            p.join()

    def transform(self, batch=1000):
        if batch != 0:
            for batch_size in range(0+batch,self.data.shape[0],batch):
                start_chunk = 0 if batch_size == batch else batch
                end_chunk = batch_size
                for n,url in tqdm(enumerate(self.data[0][start_chunk:end_chunk])):
                    if str(n)+'.png' in os.listdir(self.path_2_dataset):
                        continue
                    try:
                        response = requests.get(url)
                    except Exception as _ConnectionError:
                        print('Error getting ' + str(n) + ' image. Full log:'+str(_ConnectionError))
                        continue 

                    try:
                        img = Image.open(BytesIO(response.content))
                        img.save(self.path_2_dataset+'/'+str(n)+'.png')
                    except IOError:

                        try:
                            img.convert('RGB').save(self.path_2_dataset+'/'+str(n)+'.png',optimize=True)
                        except Exception as GenericError:
                            print('Error saving the ' + str(n) + ' image. Full log:'+str(GenericError))

                    time.sleep(np.random.uniform(0,1))
        else:
            for n,url in tqdm(enumerate(self.data[0])):
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img.convert('RGB').save(self.path_2_dataset+'/'+str(n)+'.png')


if __name__ == '__main__':

    path_2_file = './../data/'
    path_2_dataset = '/media/mazzola/TOSHIBA EXT/dataset/meme'

    loader = Loader(path_2_file=path_2_file, 
                    path_2_dataset=path_2_dataset)
    loader.fit()
    loader.run_multiprocess_download(16)
    
    ###TESTING FUNCTIONS###
    #loader.transform(batch=100_000)
    #loader.download_batch(1,10,100)
    #loader.run_multiprocess_download(16)