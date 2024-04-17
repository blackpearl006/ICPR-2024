'''Code snippet to check the Inter-Subject interactions pertaining to a specific ROI.'''
import os
import gc
import persim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import Rips
import warnings
warnings.filterwarnings("ignore")

# networks = ['default_mode','cerebellum','frontoparietal','occipital','cingulo-opercular', 'sensorimotor']
networks = [ 'DMN','CB','FP','OP','CO','SM']
roi_count = [34,18,21,22,32,33]
descriptors = ['H0','H1','H2']

def get_data(network):
    data = {}
    for subject_class in ['HC','MCI']:
        data_dir = f'path/to/{network}/{subject_class}'
        for j in os.listdir(data_dir):
            df = pd.read_csv(os.path.join(data_dir,j), header=None)
            if not (df.values == 0).any(axis=1).any():
                data[f'{subject_class}_{j}'] = np.array(df.values)
    return data

def persis_plot(df, tau=1, dim = 3):
    df = df.reshape(-1,1)
    emb_vector = [[ df[i+j*tau][0] for j in range(dim)] for i in range(len(df) - dim*tau)]

    rips = Rips(maxdim=2)
    dgms = rips.fit_transform(np.array(emb_vector))
    H0_dgm = dgms[0]
    H1_dgm = dgms[1]
    H2_dgm = dgms[2]

    return (H0_dgm,H1_dgm,H2_dgm)

Betti_matrix = {}
for network,rois in zip(networks,roi_count):
    Betti_matrix={}
    data = get_data(network=network)
    for sub_name in list(data.keys()):
        Betti_matrix[sub_name] = {}
        for roi in range(rois):
            Betti_matrix[sub_name][roi] = {}
            H0_dgm,H1_dgm,H2_dgm=np.array([]),np.array([]),np.array([])
            
            H0_dgm,H1_dgm,H2_dgm = persis_plot(data[sub_name][roi])
            Betti_matrix[sub_name][roi]['H0']=H0_dgm
            Betti_matrix[sub_name][roi]['H1']=H1_dgm
            Betti_matrix[sub_name][roi]['H2']=H2_dgm
            print('Betti descriptors calculated !!')

    for roi in range(rois):
        dist1, dist2, dist3 = [], [], []
        data = get_data(network=network)
        class_dir = f'outpath/ROIanalysis/{network}/'
        os.makedirs(class_dir,exist_ok=True)
        for sub_number in range(len(list(data.keys()))):
            sub_list1,sub_list2,sub_list3=[],[],[]
            sub_name = list(data.keys())[sub_number]
            sub1_h1 = Betti_matrix[sub_name][roi]['H0']
            sub1_h2 = Betti_matrix[sub_name][roi]['H1']
            sub1_h3 = Betti_matrix[sub_name][roi]['H2']
            for sub_number2 in range(len(list(data.keys()))):
                sub_name2 = list(data.keys())[sub_number2]
                sub2_h1 = Betti_matrix[sub_name2][roi]['H0']
                sub2_h2 = Betti_matrix[sub_name2][roi]['H1']
                sub2_h3 = Betti_matrix[sub_name2][roi]['H2']
                sub_list1.append(persim.wasserstein(sub1_h1,sub2_h1,matching=False))
                sub_list2.append(persim.wasserstein(sub1_h2,sub2_h2,matching=False))
                sub_list3.append(persim.wasserstein(sub1_h3,sub2_h3,matching=False))
            
            dist1.append(sub_list1)
            dist2.append(sub_list2)
            dist3.append(sub_list3)
        for num, dist_img in enumerate([dist1,dist2,dist3]):
            plt.imshow(dist_img)
            plt.savefig(f'{class_dir}{roi}_h{num}.png',dpi=200)
            plt.close()
            gc.collect()