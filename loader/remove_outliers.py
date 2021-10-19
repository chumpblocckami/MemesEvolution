import multiprocessing
import os 
from PIL import Image
import numpy as np 
import shutil 
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
import time 
import logging 
logging.basicConfig(filename='./../memes_preprocessing.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


class Preprocessor():
    def __init__(self, path_2_dataset, path_2_outliers):
        self.logger = logging.getLogger('Preprocessing')

        self.data_path = path_2_dataset
        self.outlier_path = path_2_outliers
        self.memes = os.listdir(path_2_dataset)

    def remove_outliers(self, process, lower_bound, upper_bound):
        for img_url in tqdm(self.memes[lower_bound:upper_bound],desc='Process '+str(process)+' is preprocessing...'):
            try:
                target = Image.open(self.data_path+'/'+img_url)
                pixels = dict(Counter(np.asarray(target).reshape(1,-1)[0]))
                if 0 not in pixels.keys():
                    self.logger.info('Meme'+str(img_url)+' is good')
                    continue
                if len(pixels.keys()) > 2:
                    self.logger.info('meme'+str(img_url)+' has more than 2 colors.')
                    continue
                if pixels[0] > sum(value for key,value in pixels.items() if key != 0):
                    shutil.move(self.data_path+'/'+img_url, self.outlier_path+'/'+img_url)
            except Exception as generic_exception:
                self.logger.error('Cannot process image'+str(img_url)+'./tFull log: '+str(generic_exception))
            time.sleep(np.random.uniform(0,1))
    
    def run_multiprocess_preprocessing(self,n_instances):
        n_memes = len(self.memes)
        batches_size = range(0,n_memes,int(n_memes/n_instances))
        self.logger.info('Preprocessing '+str(n_memes)+' memes using '+str(n_instances)+' istances.')
        excess = (n_memes - max(batches_size))
        processes = []
        for pid in range(n_instances):
            lower = batches_size[pid]
            upper = batches_size[pid+1]-1 + excess if pid == n_instances else batches_size[pid+1]-1
            processes.append(
                mp.Process(target = self.remove_outliers,
                args=(pid, lower, upper))
                )

        for p in processes:
            p.start()
            time.sleep(1)
            
        for p in processes:
            p.join()

if __name__ == '__main__':
    path_2_dataset = '/media/mazzola/TOSHIBA EXT/dataset/meme'
    path_2_outliers = '/media/mazzola/TOSHIBA EXT/dataset/meme_outlier'

    preprocessor = Preprocessor(path_2_dataset=path_2_dataset,
                                path_2_outliers=path_2_outliers)
    preprocessor.run_multiprocess_preprocessing(8)
    #preprocessor.remove_outliers(1,0,100)