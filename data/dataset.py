import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import random

#SAVE_DIR = 'C:/dataset/Spatial_single/'
SAVE_DIR = 'C:/dataset/Spatial/'
#SAVE_DIR = '/home/cvml/spatial/dataset/Spatial/'
IMAGE_DIR = 'D:/dataset/ShapeNetRendering/'
POINT_DIR = 'D:/dataset/ShapeNetPointcloud/'
OBJ_DIR = 'D:/dataset/ShapeNetCore.v2/'
CATEGORY_FILE = './data/synsetoffset2category.txt'

class SpatialDataset(data.Dataset):
    def __init__(self, class_choice=None, train=True):
        self.class_choice = class_choice
        self.train = train

        self.spatialData = sorted(os.listdir(SAVE_DIR))
        if not class_choice is None:
            self.spatialData = [path for path in self.spatialData if path.split('_')[1] == class_choice]

        if self.train:
            #self.spatialData = self.spatialData[:int(len(self.spatialData)*0.8)]
            self.spatialData = self.spatialData[:960]
        else:
            self.spatialData = self.spatialData[int(len(self.spatialData)*0.8):]

        self.augmentation_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(size=200, interpolation=Image.BILINEAR),
                                        transforms.Pad(padding=20, fill=(255,255,255), padding_mode='constant'),
                                        transforms.RandomCrop(size=224),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()])

    def __getitem__(self, index):
        fn = os.path.join(SAVE_DIR, self.spatialData[index])
        _, point, _ = torch.load(fn) # {[Img_Tensor, Img_Tensor, ...]}, {PointCloud_Tensor}, {AdjacencyMatrix_SparseTensor}

        #vertex_num = adj[2][0]

        #imgs = self.augmentation(imgs)
        
        #adj = torch.sparse.FloatTensor(adj[0], adj[1].type(torch.float), adj[2])
        
        #laplacian, _ = self.getLaplacian(adj)
        
        #return imgs, laplacian, point
        #point = point + torch.ones(1,3).type(point.type())
        return point

    def __len__(self):
        return len(self.spatialData)

    def getLaplacian(self, adjacency):
        vertex_num = adjacency.size()[0]
        adj_dense = adjacency.to_dense()
        normalized_degree = torch.zeros(vertex_num, vertex_num)

        for vertex in range(vertex_num):
            normalized_degree[vertex, vertex] = torch.sqrt(torch.sum(adj_dense[vertex]))

        symmetric_normalized_laplacian = torch.mm(torch.mm(normalized_degree, adj_dense), normalized_degree)
        
        return symmetric_normalized_laplacian, normalized_degree

    def augmentation(self, imgs):
        imgs = [imgs[inx] for inx in range(imgs.size()[0])]

        imgs_aug = []
        for img in imgs:
            img = self.augmentation_transform(img)
            imgs_aug.append(img)
        
        imgs_aug = torch.stack(imgs_aug)
        return imgs_aug




class ShapeNetDataset(data.Dataset):
    def __init__(self, class_choice=None):
        self.class_choice = class_choice

        self.categories = {}
        self.paired_data = {}

        self.loadData(class_choice=self.class_choice)
        print('Start Spatial dataset save...')
        self.savePT(save_dir=SAVE_DIR)
        print('Finish Spatial dataset save...')

        
    def loadData(self, class_choice=None):
        # Category load    
        with open(CATEGORY_FILE, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.categories[ls[0]] = ls[1]
            if not self.class_choice is None:
                self.categories = {k:v for k,v in self.categories.items() if k in class_choice}
            print(self.categories)

        # {Image data, Point cloud data, Mesh data} load
        """
        img_dir   : 'D:/dataset/ShapeNetRendering/{02691156}'
        img_path  : 'D:/dataset/ShapeNetRendering/{02691156}/{1a04e3eab45ca15dd86060f189eb133}'

        point_dir : 'D:/dataset/ShapeNetPointcloud/{02691156}/ply'
        point_path: 'D:/dataset/ShapeNetPointcloud/{02691156}/ply/{1a04e3eab45ca15dd86060f189eb133}.points.ply'

        obj_dir   : 'D:/dataset/ShapeNetCore.v2/{02933112}'
        obj_path  : 'D:/dataset/ShapeNetCore.v2/{02933112}/{1a04e3eab45ca15dd86060f189eb133}/models/model_normalized.obj'
        """
        empty = []
        for category in self.categories:
            img_dir = os.path.join(IMAGE_DIR, self.categories[category])
            img_list = sorted(os.listdir(img_dir))

            try:
                point_dir = os.path.join(POINT_DIR, self.categories[category], 'ply')
                point_list = sorted(os.listdir(point_dir))
            except:
                point_list = []
            
            paired_img = [img_path for img_path in img_list if img_path + '.points.ply' in point_list]
            print('category ', self.categories[category], 'files ' + str(len(paired_img)) + ' / ' + str(len(img_list)))

            if len(paired_img) != 0:
                self.paired_data[category] = []
                for paired_path in paired_img:
                    obj_path = OBJ_DIR + self.categories[category] + '/' + paired_path + '/models/model_normalized.obj'
                    self.paired_data[category].append((os.path.join(img_dir, paired_path, 'rendering'), os.path.join(point_dir, paired_path + '.points.ply'), obj_path, category))
            else:
                empty.append(category)
        for category in empty:
            del self.categories[category]


    def savePT(self, save_dir):
        save_count = 0
        for category in self.categories:
            length = len(self.paired_data[category])
            printProgressBar(0, length, prefix=category, suffix='0 saved.', length=40)
            for data_inx, (imgs_path, point_path, obj_path, _) in enumerate(self.paired_data[category]):

                imgs = self.loadImg(imgs_path) # (img_num, 137, 137, 3)
                point = self.loadPoint(point_path, point_num=3000) # (vertex_num, 3)
                adj = self.loadAdj(obj_path) # (Tensor(edge_num, 2), Tensor(edge_num,), Tuple(vertex_num, vertex_num)) 

                vertex_num = adj[2][0]
                if vertex_num < 3000:
                    save_count += 1
                    torch.save((imgs, point, adj), save_dir+'spatial_'+category+'_'+str(data_inx)+'.pt')

                printProgressBar(data_inx+1, length, prefix=category, suffix='{} fils saved.'.format(save_count), length=40)


    def loadImg(self, path):
        imgs_path = sorted(os.listdir(path))
        imgs_path = [img_path for img_path in imgs_path if img_path.split('.')[-1] == 'png']
        imgs = []
        background = Image.new('RGBA', (137,137), (255,255,255))

        for img_path in imgs_path:
            png = Image.open(os.path.join(path,img_path))
            rgba = Image.alpha_composite(background, png)

            img = rgba.convert('RGB')
            img = transforms.ToTensor()(img)
            imgs.append(img)

        imgs = torch.stack(imgs)

        return imgs


    def loadPoint(self, path, point_num=None):
        f = open(path, mode='rt', encoding='latin-1')
        vertex_num = 0
        vertices = []
        header = True

        for inx, line in enumerate(f):
            line = line.split()

            if len(line) > 0:
                if header:
                    if line[0] == 'element' and line[1] == 'vertex':
                        vertex_num = int(line[2])     
                    if line[0] == 'end_header':
                        header = False
                else:
                    vertices.append((float(line[0]), float(line[1]), float(line[2])))

        if point_num == None:
            point_cloud = vertices
        elif vertex_num > point_num:
            point_cloud = random.sample(vertices, point_num)

        point_cloud = np.array(point_cloud)
        point_cloud = torch.from_numpy(point_cloud)

        return point_cloud


    def loadAdj(self, path):
        f = open(path, mode='rt', encoding='UTF8')
        vertex_num = 0
        degree = []
        edges = []
        header = True

        for inx, line in enumerate(f):
            line = line.split()

            if len(line) > 0:
                if header:
                    if line[0] == '#' and line[2] == 'vertex' and line[3] == 'positions':
                        vertex_num = int(line[1])
                        degree = torch.IntTensor(vertex_num)
                        header = False
                else:
                    if line[0] == 'f':
                        # Link
                        v1 = int(line[1].split('/')[0])-1
                        v2 = int(line[2].split('/')[0])-1
                        v3 = int(line[3].split('/')[0])-1
                        # Upper part
                        edges.append((v1, v2))
                        edges.append((v1, v3))
                        edges.append((v2, v3))
                        # Symmetric part
                        edges.append((v2, v1))
                        edges.append((v3, v1))
                        edges.append((v3, v2))

                    elif line[0] == 'l':
                        # Link
                        v1 = int(line[1])-1
                        v2 = int(line[2])-1
                        # Upper part
                        edges.append((v1, v2))
                        # Symmetric part
                        edges.append((v2, v1))

        adj_inx = torch.LongTensor(list(set(edges)))
        adj_value = torch.ones((len(adj_inx),), dtype=torch.uint8)
        adj_size = torch.Size([vertex_num, vertex_num])
        # adjacency = torch.sparse.FloatTensor(adj_inx.t(), adj_value, adj_size)
        adjacency = (adj_inx.t(), adj_value, adj_size)

        return adjacency


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


if __name__ == '__main__':
    ShapeNetDataset(class_choice='chair')
