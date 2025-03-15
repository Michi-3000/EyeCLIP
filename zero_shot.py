import os
import eyeclip
import torch
#from torchvision.datasets import CIFAR100
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
import torch.nn.functional as F
import numpy as np
import csv

def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        #print(cm1)
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_

class CustomImageFolder(Dataset):
    def __init__(self, dt, transform=None):
        self.dt = dt
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        images = []
        for path, cls in zip(self.dt['impath'].to_list(), self.dt['class'].to_list()):
            item = (path, cls)
            images.append(item)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path
epoch = 0
ft_path = './CLIP_ft_all_key_06-30-1427'
for epoch in os.listdir(ft_path):
#for epoch in ["Image_epoch"]:
    if epoch[0]!='e':
        continue
    #if 'epoch10' not in epoch:
    #    continue
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model, preprocess = clipv2.load('ViT-B/32', device)
    model, preprocess = eyeclip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    #checkpoint = torch.load("/home/danli/workspace/pretrain/CLIP/ft_checkpoints/CLIP_ft_allresize_04-22-0105/epoch10.pt")
    #checkpoint = torch.load("/home/danli/workspace/pretrain/CLIP/ft_checkpoints/CLIP_ft_allresize_05-02-0300/epoch0.pt")
    #print(os.path.join(ft_path, epoch))
    checkpoint = torch.load(os.path.join(ft_path, epoch))
    #/home/danli/caption/CLIP/ft_checkpoints/CLIP_ft_12-06-0119/epoch9.pt
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    data_names = ['IDRiD','OCTID','PAPILA_v2','Retina','MESSIDOR2_v2','Aptos2019','Glaucoma_Fundus','OCTDL','JSIEC']
    for data_name in data_names:
        if data_name in ['IDRiD', 'MESSIDOR2_v2', 'Aptos2019']:
            texts = ['normal', 'mild diabetic retinopathy', 'moderate diabetic retinopathy', 'severe diabetic retinopathy', 'proliferative diabetic retinopathy']
        elif data_name in ['Retina']:#
            texts = ['normal', 'cataract', 'glaucoma', 'retinal disease']
        elif data_name in ['Glaucoma_Fundus']:#
            texts = ['normal', 'early glaucoma', 'advanced glaucoma']
        elif data_name in ['OCTID']:
            texts = ['normal', 'age-related macular degeneration', 'central serous chorioretinopathy', 'diabetic retinopathy', 'macula hole']
        elif data_name in ['OCTDL']:
            texts = ['age-related macular degeneration', 'diabetic macular edema', 'epiretinal membrane', 'normal', 'retinal artery occlusion', 'retinal vein occlusion', 'vitreomacular interface disease']
        elif data_name in ['PAPILA_v2']:#
            texts = ['normal', 'glaucoma', 'suspected glaucoma']
        elif data_name in ['JSIEC']:
            texts = ['normal', 'tessellated fundus', 'large optic cup', 'non-referable diabetic retinopathy', 'moderate nonproliferative diabetic retinopathy', 'severe and proliferative diabetic retinopathy', 'possible glaucoma', 'optic atrophy', 'severe hypertensive retinopathy', 'disc swelling and elevation', 'dragged disc', 'congenital disc abnormality', 'retinitis pigmentosa', 'bietti crystalline dystrophy', 'peripheral retinal degeneration and break', 'myelinated nerve fiber', 'vitreous particles', 'fundus neoplasm', 'branch retinal vein occlusion', 'central retinal vein occlusion', 'massive hard exudates', 'yellow-white spots-flecks', 'cotton-wool spots', 'vessel tortuosity', 'chorioretinal atrophy,coloboma', 'preretinal hemorrhage', 'fibrosis', 'laser spots', 'silicon oil in eye', 'blur fundus', 'blur fundus with suspected proliferative diabetic retinopathy', 'retinal artery occlusion', 'rhegmatogenous retinal detachment', 'central serous chorioretinopathy', 'vogt-koyanagi-harada disease', 'maculopathy', 'epiretinal membrane', 'macular hole', 'pathological myopia']
        '''
        if data_name in ['IDRiD', 'MESSIDOR2', 'MESSIDOR2_v2', 'Aptos2019']:
                #texts = ['no DR', 'mild DR', 'moderate DR', 'severe DR', 'proliferative DR']
                texts = ['normal', 'mild diabetic retinopathy', 'moderate diabetic retinopathy', 'severe diabetic retinopathy', 'proliferative diabetic retinopathy']
                id_map = {"no DR":0, "mild DR":1, "moderate DR":2, "severe DR":3, "proliferative DR":4}
        elif data_name in ['Retina']:
                texts = ['normal', 'cataract', 'glaucoma', 'retina disease']
                id_map = {'normal':0, 'cataract':1, 'glaucoma':2, 'retina disease':3}
        elif data_name in ['Glaucoma_Fundus']:
                texts = ['normal', 'early glaucoma', 'advanced glaucoma']
                id_map = {'normal control':0, 'early glaucoma':1, 'advanced glaucoma':2}
        elif data_name in ['OCTID']:
                #texts = ['normal', 'AMRD', 'CSR', 'DR', 'macula hole']
                texts = ['normal', 'age-related macular degeneration', 'central serous chorioretinopathy', 'diabetic retinopathy', 'macula hole']
                id_map = {'normal':0, 'AMRD':1, 'CSR':2, 'DR':3, 'macula hole':4}
        elif data_name in ['OCTDL']:
                #texts = ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']
                texts = ['age-related macular degeneration', 'diabetic macular edema', 'epiretinal membrane', 'normal', 'retinal artery occlusion', 'retinal vein occlusion', 'vitreomacular interface disease']
                id_map = {'AMD':0, 'DME':1, 'ERM':2, 'NO':3, 'RAO':4, 'RVO':5, 'VID':6}
        elif data_name in ['PAPILA', 'PAPILA_v2']:
                texts = ['normal', 'glaucoma', 'suspected suspecious glaucoma']
        elif data_name in ['JSIEC']:
                texts = ['normal', 'tessellated fundus', 'large optic cup', 'non-referable diabetic retinopathy', 'moderate nonproliferative diabetic retinopathy', 'severe and proliferative diabetic retinopathy', 'possible glaucoma', 'optic atrophy', 'severe hypertensive retinopathy', 'disc swelling and elevation', 'dragged disc', 'congenital disc abnormality', 'retinitis pigmentosa', 'bietti crystalline dystrophy', 'peripheral retinal degeneration and break', 'myelinated nerve fiber', 'vitreous particles', 'fundus neoplasm', 'branch retinal vein occlusion', 'central retinal vein occlusion', 'massive hard exudates', 'yellow-white spots-flecks', 'cotton-wool spots', 'vessel tortuosity', 'chorioretinal atrophy-coloboma', 'preretinal hemorrhage', 'fibrosis', 'laser spots', 'silicon oil in eye', 'blur fundus without proliferative diabetic retinopathy', 'blur fundus with suspected proliferative diabetic retinopathy', 'retinal artery occlusion', 'rhegmatogenous retinal detachment', 'central serous chorioretinopathy', 'vogt-koyanagi-harada disease', 'maculopathy', 'epiretinal membrane', 'macular hole', 'pathological myopia']
        '''
        #0 stands for healthy, 1 for glaucoma, and 2 for suspicious


        num_class = len(texts)
        prediction_decode_list = []
        prediction_list = []
        true_label_decode_list = []
        true_label_onehot_list = []


        dt = pd.read_csv("data/"+data_name+'.csv')#[:20]
        text_info = eyeclip.tokenize(texts).to(device)

        dataset_test = CustomImageFolder(dt=dt, transform=preprocess)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=1,
                drop_last=False
            )

        text_features = model.encode_text(text_info).to(device)
        text_features  /= text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            for batch in data_loader_test:
                images = batch[0].to(device)
                target = batch[1].to(device)
                #print(images.shape)
                #print(target)
                true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(texts))

                combined = list(zip(indices, values))
                sorted_combined = sorted(combined, key=lambda x: x[0])

                sorted_values = [x[1].cpu().detach().item() for x in sorted_combined]
                #print(sorted_values)

                _,true_label_decode = torch.max(true_label, 1)
                true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
                true_label_onehot_list.extend(true_label.cpu().detach().numpy())
                prediction_decode_list.extend([indices[0].cpu().detach().item()])
                prediction_list.extend([sorted_values])


                # Print the result
                #print("\nTop predictions:\n")
                #for value, index in zip(values, indices):
                #    print(f"{index}: {100 * value.item():.2f}%")

        #print("prediction_decode_list:", prediction_decode_list)
        #print("true_label_decode_list:", true_label_decode_list)
        confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
        #print("confusion_matrix:", confusion_matrix)
        acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)

        #print(acc, sensitivity, specificity, precision, G, F1, mcc)

        auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
        auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')
        print(auc_roc, auc_pr)
        results_path = os.path.join(ft_path, 'metrics_{}_new.csv'.format(epoch))
        file_empty = False
        if not os.path.exists(results_path):
            file_empty = True
        with open(results_path, mode='a',newline='',encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            if file_empty:
                    header = ['dataset', 'acc', 'sensitivity', 'specificity', 'precision', 'auc_roc', 'auc_pr', 'F1', 'mcc']
                    wf.writerow(header)
            data2=[[data_name, acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc]]
            for i in data2:
                    wf.writerow(i)
    torch.cuda.empty_cache()
