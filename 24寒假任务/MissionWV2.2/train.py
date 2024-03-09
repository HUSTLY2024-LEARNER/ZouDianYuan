from ultralytics import YOLO



import os, shutil, random
import numpy as np

def split_data(data_path,label_path,des_path,val_size=0.1, test_size=0.2,postfix = 'jpg'):

    dataset_path = des_path

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(f'{dataset_path}/images', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/val', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/test', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/val', exist_ok=True) 
    os.makedirs(f'{dataset_path}/labels/test', exist_ok=True)

    path_list = np.array([i.split('.')[0] for i in os.listdir(label_path) if 'txt' in i])
    random.shuffle(path_list)
    train_id, val_id, test_id = np.split(path_list, [int((1-val_size-test_size)*len(path_list)), int((1-test_size)*len(path_list))])

    idx = 0
    for (i, id) in enumerate(train_id):
        shutil.copy(f'{data_path}/{id}.{postfix}', f'{dataset_path}/images/train/{idx:06d}.{postfix}')
        shutil.copy(f'{label_path}/{id}.txt', f'{dataset_path}/labels/train/{idx:06d}.txt')
        idx += 1
    
    for (i, id) in enumerate(val_id):
        shutil.copy(f'{data_path}/{id}.{postfix}', f'{dataset_path}/images/val/{idx:06d}.{postfix}')
        shutil.copy(f'{label_path}/{id}.txt', f'{dataset_path}/labels/val/{idx:06d}.txt')
        idx += 1
    
    for (i, id) in enumerate(test_id):
        shutil.copy(f'{data_path}/{id}.{postfix}', f'{dataset_path}/images/test/{idx:06d}.{postfix}')
        shutil.copy(f'{label_path}/{id}.txt', f'{dataset_path}/labels/test/{idx:06d}.txt')
        idx += 1        

# def split_data(data_path,label_path,des_path,val_size=0.1, test_size=0.2,postfix = 'jpg'):
    
#     dataset_path = des_path
    
#     os.makedirs(dataset_path, exist_ok=True)
#     os.makedirs(f'{dataset_path}/images', exist_ok=True)
#     os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
#     os.makedirs(f'{dataset_path}/images/val', exist_ok=True)
#     os.makedirs(f'{dataset_path}/images/test', exist_ok=True)
#     os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
#     os.makedirs(f'{dataset_path}/labels/val', exist_ok=True)
#     os.makedirs(f'{dataset_path}/labels/test', exist_ok=True)
    
#     path_list = np.array([i.split('.')[0] for i in os.listdir(label_path) if 'txt' in i])
#     random.shuffle(path_list)
#     train_id = path_list[:int(len(path_list) * (1 - val_size - test_size))]
#     val_id = path_list[int(len(path_list) * (1 - val_size - test_size)):int(len(path_list) * (1 - test_size))]
#     test_id = path_list[int(len(path_list) * (1 - test_size)):]
    
#     for i in train_id:
#         shutil.copy(f'{data_path}/{i}.{postfix}', f'{dataset_path}/images/train/{i}.{postfix}')
#         shutil.copy(f'{label_path}/{i}.txt', f'{dataset_path}/labels/train/{i}.txt')
    
#     for i in val_id:
#         shutil.copy(f'{data_path}/{i}.{postfix}', f'{dataset_path}/images/val/{i}.{postfix}')
#         shutil.copy(f'{label_path}/{i}.txt', f'{dataset_path}/labels/val/{i}.txt')
    
#     for i in test_id:
#         shutil.copy(f'{data_path}/{i}.{postfix}', f'{dataset_path}/images/test/{i}.{postfix}')
#         shutil.copy(f'{label_path}/{i}.txt', f'{dataset_path}/labels/test/{i}.txt')

if __name__ == '__main__':
    #  split_data(data_path="D:/Winter_yolo/full_field4/images",label_path="D:/Winter_yolo/full_field4/labels",
    #             des_path="./data",val_size=0.1,test_size=0.2,postfix='jpg')
    # Load a model
    # model = YOLO("pretrain_model/yolov9c.pt", task = 'detect')

    model = YOLO("D:\Desktop\py-program\RoboMaster_YOLO\RMWinter\RMWinter_YOLO12\weights\last.pt",task='detect')

    # https://blog.csdn.net/qq_37553692/article/details/130898732#10_workers__187
    # said cannot use multi workers to load data
    # Train
    # batch = 16 epoch=300
    # worker=6 can but slow (full gpu 1min12s) (train s_model)
    # worker=8 can and quick (full gpu 10s) (train n_model)
    # crash when close dataloader mosanic
    # 20G disk space need
    # demand c disk which I havn't considered


    model.train(data="dataset.yaml", workers=6, epochs=50, batch = 16, cfg="cfg.yaml") 
    model.val(split="test")