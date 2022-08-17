import os
import numpy as np
import cv2

class DataConfig(object):
    def __init__(self, fs):
        self.fs = fs
        self.data_dir = '/abc'
        self.split_dir = '/def'
        self.train_files = [
            'scene1 cam1 1',
            'scene2 cam2 2',
            'scene3 cam2 3',
            'scene4 cam4 4',
            'scene5 cam5 5',
        ]
        self.test_files = [
            'scene6 cam6 6',
            'scene7 cam7 7',
            'scene8 cam8 8'
        ]

        self.fname_metadata_map = {
            'RGB': ('.jpg', (1024, 1024, 3), 255), 
            'DEPTH': ('.png', (1024, 1024), 65535),
            'semantics': ('.png', (1024, 1024), 150),
            'pom': ('.png', (128, 128), 2),
            'partial_occ': ('.png', (128, 128), 255),
        }

        for fname, (ext, img_dim, img_range) in self.fname_metadata_map.items():
            for files in [self.train_files, self.test_files]:
                for row in files:
                    scene, camera, fileidx = row.split()
                    fp = os.path.join(self.data_dir, scene, '0', camera, fname, f'{fileidx}{ext}')
                    img = np.random.rand(*img_dim) * img_range
                    img = img.astype(np.uint8 if img_range < 256 else np.uint16)
                    img = np.array(cv2.imencode(ext, img)[1]).tobytes()
                    self.fs.create_file(fp, contents=img)

        self.fs.create_file(os.path.join(self.split_dir, 'train.txt'), \
            contents='\n'.join(self.train_files))
        
        self.fs.create_file(os.path.join(self.split_dir, 'test.txt'), \
            contents='\n'.join(self.test_files))
