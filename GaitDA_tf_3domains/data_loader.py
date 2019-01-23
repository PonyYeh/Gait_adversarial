import scipy
from glob import glob
import numpy as np
import cv2 
import os
import random 
random.seed(1337)
np.random.seed(1337)  # for reproducibility
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

IMG_RES = (160, 48,1)
RATIO = (62,62)
class DataLoader_Random():
    def __init__(self):
        self.ratio = RATIO
        self.img_res = IMG_RES
        self.data_dir =  "../GaitRecognition/DatasetB_GEI"
        
    def load_img(self,path):
        gray_img = cv2.imread(path,0) # shape(160,60)
#         rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB) #shape(160,60,3)
        img = cv2.resize(gray_img,(self.img_res[1],self.img_res[0])) 
        img = img.reshape((self.img_res[0], self.img_res[1], self.img_res[2]))
        return img
    
    def getRandomData(self, batches, if_train=True):
        ids = np.arange(0,self.ratio[0])
        condition_name = ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        angle_name = ['000','018','036','054','072','090','108','126','144','162','180']
        
        
        select_cond = []
        train_cond = [0,4]
        test_cond = [1,5]
        if(if_train):
            select_cond =  train_cond
        else:
            select_cond = test_cond 
        
        n_person = len(ids)-1
        n_cond = len(condition_name)-1
        n_ang = len(angle_name)-1
        
        id_list = []
        cond_list = []
        ang_list = []
        
        pairs = []
        same_pair = []
        domain_labels = []
        
        for i in range(batches):
            while True:
                id1_ = random.randint(0,n_person)
                id1 = "%03d" % id1_
#                 cond1_ = random.randint(0,n_cond)
                cond1_ = random.choice(select_cond)
                cond1 = condition_name[cond1_]
                ang1_ = random.randint(0,n_ang)
                ang1 = angle_name[ang1_]
                path1 = self.data_dir +'/'+ id1+'/'+cond1+'/'+ang1+'/'+id1+'-'+cond1+'-'+ang1 +'.jpg'
                if os.path.exists(path1):
                    break

            while True:
                id2 = 0
                if i %2 ==1:
                    id2_ =id1_
                else:
                    id2_ = random.randint(0,n_person)
                    while id2_==id1_:
                        id2_ = random.randint(0,n_person)
                id2 = "%03d" % id2_
#                 cond2_ = random.randint(4,n_cond)
#                 cond2_ = random.randint(0,n_cond)
                cond2_ = random.choice(select_cond)
                cond2 = condition_name[cond2_]
                ang2_ = random.randint(0,n_ang)
                ang2 = angle_name[ang2_]
                path2 = self.data_dir +'/'+ id2+'/'+cond2+'/'+ang2+'/'+id2+'-'+cond2+'-'+ang2 +'.jpg'
                if os.path.exists(path2):
                    break
            print("path",path1,path2)  
            id_list += [[id1_,id2_]]
            cond_list+= [[cond1_,cond2_]]
            ang_list+= [[ang1_,ang2_]]
            
            if_target = lambda con : 1 if((con>3)*(con<10)) else 0  #target 90 & nm
#             if_target = lambda ang,con : 1 if((ang==5) & ((con>3)*(con<10))) else 0  #target 90 & nm
            domain_labels += [[if_target(cond1_),if_target(cond2_)]]
#             domain_labels += [[if_target(ang1_,cond1_),if_target(ang2_,cond2_)]]
             
            img1 = self.load_img(path1)   
            img2 = self.load_img(path2)   
            pairs += [[img1, img2]]
            same_pair += [i%2]
            
        pairs_label = np.dstack([id_list, cond_list,ang_list])
        pairs= np.array(pairs)/255.0
        pairs = pairs.astype('float32')
        pairs0 = pairs[:,0,:,:,:]
        pairs1 = pairs[:,1,:,:,:]
        same_pair = np.array(same_pair) 
        domain_labels = np.array(domain_labels) 
        return pairs0, pairs1, same_pair, pairs_label, domain_labels
    
    def getBalanceData(self, batches):
        ids = np.arange(0,self.ratio[0])
        condition_name = ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        angle_name = ['000','018','036','054','072','090','108','126','144','162','180']
        
        n_person = len(ids)-1
        n_cond = len(condition_name)-1
        n_ang = len(angle_name)-1
#         batches = 10000
        
        id_list = []
        cond_list = []
        ang_list = []
        
        pairs = []
        same_pair = []
        domain_labels = []
        line = int(batches/4)
        for i in range(batches):
            if (i<line):
                print("ST",i)
                while True:
                    id1_ = random.randint(0,n_person)
                    id1 = "%03d" % id1_
                    cond1_ = random.randint(0,1)
    #                 cond1_ = random.randint(0,n_cond)
                    cond1 = condition_name[cond1_]
                    ang1_ = random.randint(0,n_ang)
#                     ang1_ = 5
                    ang1 = angle_name[ang1_]
                    path1 = self.data_dir +'/'+ id1+'/'+cond1+'/'+ang1+'/'+id1+'-'+cond1+'-'+ang1 +'.jpg'
                    if os.path.exists(path1):
                        break

                while True:
                    id2 = 0
                    if i %2 ==1:
                        id2_ =id1_
                    else:
                        id2_ = random.randint(0,n_person)
    #                     id2 = ids[id2] 
                        while id2_==id1_:
                            id2_ = random.randint(0,n_person)
                    id2 = "%03d" % id2_
                    cond2_ = random.randint(4,n_cond)
    #                 cond2_ = random.randint(0,n_cond)
                    cond2 = condition_name[cond2_]
#                     ang2_ = 5
                    ang2_ = random.randint(0,n_ang)
                    ang2 = angle_name[ang2_]
                    path2 = self.data_dir +'/'+ id2+'/'+cond2+'/'+ang2+'/'+id2+'-'+cond2+'-'+ang2 +'.jpg'
                    if os.path.exists(path2):
                        break
            elif(i>=line and i<(2*line)):
                print("TS",i)
                while True:
                    id1_ = random.randint(0,n_person)
                    id1 = "%03d" % id1_
                    cond1_ = random.randint(4,n_cond)
    #                 cond2_ = random.randint(0,n_cond)
                    cond1 = condition_name[cond1_]
#                     ang1_ = 5
                    ang1_ = random.randint(0,n_ang)
                    ang1 = angle_name[ang1_]
                    path1 = self.data_dir +'/'+ id1+'/'+cond1+'/'+ang1+'/'+id1+'-'+cond1+'-'+ang1 +'.jpg'
                    if os.path.exists(path1):
                        break

                while True:
                    id2 = 0
                    if i %2 ==1:
                        id2_ =id1_
                    else:
                        id2_ = random.randint(0,n_person)
    #                     id2 = ids[id2] 
                        while id2_==id1_:
                            id2_ = random.randint(0,n_person)
                    id2 = "%03d" % id2_
                    cond2_ = random.randint(0,1)
    #                 cond2_ = random.randint(0,n_cond)
                    cond2 = condition_name[cond2_]
                    ang2_ = random.randint(0,n_ang)
#                     ang2_ = 5
                    ang2 = angle_name[ang2_]
                    path2 = self.data_dir +'/'+ id2+'/'+cond2+'/'+ang2+'/'+id2+'-'+cond2+'-'+ang2 +'.jpg'
                    if os.path.exists(path2):
                        break
            elif(i>=(2*line) and i<(3*line)):
                print("TT",i)
                while True:
                    id1_ = random.randint(0,n_person)
                    id1 = "%03d" % id1_
                    cond1_ = random.randint(4,n_cond)
    #                 cond2_ = random.randint(0,n_cond)
                    cond1 = condition_name[cond1_]
                    ang1_ = random.randint(0,n_ang)
#                     ang1_ = 5
                    ang1 = angle_name[ang1_]
                    path1 = self.data_dir +'/'+ id1+'/'+cond1+'/'+ang1+'/'+id1+'-'+cond1+'-'+ang1 +'.jpg'
                    if os.path.exists(path1):
                        break

                while True:
                    id2 = 0
                    if i %2 ==1:
                        id2_ =id1_
                    else:
                        id2_ = random.randint(0,n_person)
    #                     id2 = ids[id2] 
                        while id2_==id1_:
                            id2_ = random.randint(0,n_person)
                    id2 = "%03d" % id2_
                    cond2_ = random.randint(4,n_cond)
    #                 cond2_ = random.randint(0,n_cond)
                    cond2 = condition_name[cond2_]
#                     ang2_ = 5
                    ang2_ = random.randint(0,n_ang)
                    ang2 = angle_name[ang2_]
                    path2 = self.data_dir +'/'+ id2+'/'+cond2+'/'+ang2+'/'+id2+'-'+cond2+'-'+ang2 +'.jpg'
                    if os.path.exists(path2):
                        break 
            else:
                print("SS",i)
                while True:
                    id1_ = random.randint(0,n_person)
                    id1 = "%03d" % id1_
                    cond1_ = random.randint(0,1)
    #                 cond2_ = random.randint(0,n_cond)
                    cond1 = condition_name[cond1_]
                    ang1_ = random.randint(0,n_ang)
#                     ang1_ = 5
                    ang1 = angle_name[ang1_]
                    path1 = self.data_dir +'/'+ id1+'/'+cond1+'/'+ang1+'/'+id1+'-'+cond1+'-'+ang1 +'.jpg'
                    if os.path.exists(path1):
                        break

                while True:
                    id2 = 0
                    if i %2 ==1:
                        id2_ =id1_
                    else:
                        id2_ = random.randint(0,n_person)
    #                     id2 = ids[id2] 
                        while id2_==id1_:
                            id2_ = random.randint(0,n_person)
                    id2 = "%03d" % id2_
                    cond2_ = random.randint(0,1)
    #                 cond2_ = random.randint(0,n_cond)
                    cond2 = condition_name[cond2_]
                    ang2_ = random.randint(0,n_ang)
#                     ang2_ = 5
                    ang2 = angle_name[ang2_]
                    path2 = self.data_dir +'/'+ id2+'/'+cond2+'/'+ang2+'/'+id2+'-'+cond2+'-'+ang2 +'.jpg'
                    if os.path.exists(path2):
                        break
            print("path",path1,path2)        
            id_list += [[id1_,id2_]]
            cond_list+= [[cond1_,cond2_]]
            ang_list+= [[ang1_,ang2_]]
            
            if_target = lambda con : 1 if((con>3)*(con<10)) else 0  #target 90 & nm
#             if_target = lambda ang,con : 1 if((ang==5) & ((con>3)*(con<10))) else 0  #target 90 & nm
            domain_labels += [[if_target(cond1_),if_target(cond2_)]]
#             domain_labels += [[if_target(ang1_,cond1_),if_target(ang2_,cond2_)]]
             
            img1 = self.load_img(path1)   
            img2 = self.load_img(path2)   
            pairs += [[img1, img2]]
            same_pair += [i%2]
            
        pairs_label = np.dstack([id_list, cond_list,ang_list])
        pairs= np.array(pairs)/255.0
        pairs = pairs.astype('float32')
        pairs0 = pairs[:,0,:,:,:]
        pairs1 = pairs[:,1,:,:,:]
        same_pair = np.array(same_pair) 
        domain_labels = np.array(domain_labels) 
        return pairs0,pairs1, same_pair, pairs_label, domain_labels
            
            
class DataLoader():
    def __init__(self):
        self.ratio = RATIO
        self.img_res = IMG_RES
        self.data_dir =  "../GaitRecognition/DatasetB_GEI"

    def load_Casia(self):
        img_list = []
        label_list = []
        angle_list = []
        condition_list = []
        num_label = 0
        img_size = self.img_res

        sorted_datasetPath = sorted(os.listdir(self.data_dir))
        print('suject nums from data: ', sorted_datasetPath )

        for person in sorted(os.listdir(self.data_dir)):
            person_path = os.path.join(self.data_dir,person)
            if ('.ipynb_checkpoints') in person:
                print("delete ipy")
                continue
            elif ("p") in person:
                print("delete ipy")
                continue
            elif ("34") in person:       #-------------revise
                print("delete 34")
                num_label += 1
                continue
            for condition in os.listdir(person_path):
                condition_path = os.path.join(person_path,condition)
                for angle in os.listdir(condition_path):      
                    angle_path = os.path.join(condition_path,angle)
                    for f in os.listdir(angle_path): 
                        print('file',f)
                        file_path = os.path.join(angle_path,f)
                        gray_img = cv2.imread(file_path,0) # shape(160,60)
#                             rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB) #shape(160,60,3)
                        img = cv2.resize(gray_img,(img_size[1],img_size[0]))  #width height
                        img_list.append(img)
                        label_list.append(num_label)
                        angle_list.append(angle)
                        condition_list.append(condition)
            num_label += 1
      
        condition_name = {'bg-01':0,'bg-02':1, 'cl-01':2, 'cl-02':3, 'nm-01':4, 'nm-02':5 , 'nm-03':6 ,'nm-04':7,'nm-05':8, 'nm-06':9}
        condition_label = []
        for item in condition_list:
            condition_label.append(condition_name[item])   
        angle_name = {'000':0, '018':1, '036':2, '054':3, '072':4, '090':5, '108':6,'126':7,'144':8,'162':9, '180':10}
        angle_label = []
        for item in angle_list:
            angle_label.append(angle_name[item])
           
        img_array = np.array(img_list) 
        label_array = np.array(label_list).reshape(-1,1)
        angle_array = np.array(angle_label).reshape(-1,1)
        condition_array = np.array(condition_label).reshape(-1,1)
        combine_label = np.concatenate((label_array, condition_array, angle_array),axis=1)

        num_angle = angle_array.max()+1 
        img_array = img_array.astype('float32')
        img_array = img_array.reshape((-1, img_size[0], img_size[1], img_size[2]))/255.0
        print("img_array", img_array.shape)
        print('num_angle', num_angle)
        print('num_label', num_label)
        print('combine_label', combine_label.shape)
        return img_array, combine_label

    def DataforTrain_Test(self,img_array, combine_label): #cyclegan unsupervised training data
        print("\n------------------start to split train/test---------------")
#         bool_train = ((combine_label[:,0] < self.ratio[0]) & (combine_label[:,1]==0 )) | ((combine_label[:,0] < self.ratio[0]) & (combine_label[:,1]==1)) | ((combine_label[:,0] < self.ratio[0]) & (combine_label[:,1]==4 )) | ((combine_label[:,0] < self.ratio[0]) & (combine_label[:,1]==5 )) 
#         bool_test = ((combine_label[:,0] >= self.ratio[0]) & (combine_label[:,1]!=2 )) & ((combine_label[:,0] >= self.ratio[0]) & (combine_label[:,1]!=3))

        bool_train = (combine_label[:,0] < self.ratio[0])  #-------------revise
        bool_test = (combine_label[:,0] >= self.ratio[0])  #-------------revise
                     
        X_train = img_array[np.where(bool_train)[0]]
        y_train = combine_label[np.where(bool_train)[0]]
        X_test = img_array[np.where(bool_test)[0]]
        y_test = combine_label[np.where(bool_test)[0]]
        
        unique, counts = np.unique(y_train[:,0], return_counts=True)
        print("y_train person",dict(zip(unique, counts)))
#         unique, counts = np.unique(y_val[:,0], return_counts=True)
#         print("y_val person",dict(zip(unique, counts)))
        unique, counts = np.unique(y_test[:,0], return_counts=True)
        print("y_test person",dict(zip(unique, counts)))
        unique, counts = np.unique(y_train[:,1], return_counts=True)
        print("y_train condition",dict(zip(unique, counts)))
#         unique, counts = np.unique(y_val[:,1], return_counts=True)
#         print("y_val condition",dict(zip(unique, counts)))
        unique, counts = np.unique(y_test[:,1], return_counts=True)
        print("y_test condition",dict(zip(unique, counts)))
        
        
        print("DataforTrain_Test",X_train.shape, y_train.shape, X_test.shape, y_test.shape) 
#         print("y_train",y_train[:50],y_train[:-50])
        return X_train, y_train, X_test, y_test
    
    def DataforSource_Target(self, X_train, y_train):
        bool_target = (y_train[:,2]== 5) & ((y_train[:,1]>3)*(y_train[:,1]<10)) #90 & nm
#         bool_source = ~((combine_label[:,1]== 6) & ((combine_label[:,2]>4)*(combine_label[:,2]<11)))
        Xtrain_T = X_train[np.where(bool_target)[0]]
        ytrain_T = y_train[np.where(bool_target)[0]]
        Xtrain_S = X_train[np.where(~bool_target)[0]]
        ytrain_S = y_train[np.where(~bool_target)[0]]
        print("DataforSource_Target",Xtrain_T.shape, ytrain_T.shape, Xtrain_S.shape, ytrain_S.shape) 
        print("y_train_T",ytrain_T[:50])
        print("y_train_S",ytrain_S[:50])
        return Xtrain_T, ytrain_T, Xtrain_S, ytrain_S

    def DataforTrainPair(self, X_train, y_train):
        print("\n------------------start to create train pair---------------")
        def create_pairs(x,y, digit_indices):
            '''Positive and negative pair creation.
            Alternates between positive and negative pairs.
            '''
            y_pairs = []
            pairs = []
            labels = []
            domain_labels = []
            num_person = len(digit_indices)
            print('num_person', num_person)
            for d in range(num_person):
                num = len(digit_indices[d])
                print(d,num)

            n = min([len(digit_indices[d]) for d in range(num_person)] )- 1
            print("create_pairs n",n)
            print("張數會等於,n*人數*2")
            
#             if_target = lambda y : 1 if((y[2]==5) & ((y[1]>3)*(y[1]<10))) else 0  #target 90 & nm
            if_domain = lambda y : 2 if((y[1]>3)*(y[1]<10)) else (0 if(y[1]<2) else 1)  #-------------revise
#             if_angle = lambda y : 2 if((y[2]>3)*(y[1]<10)) else (0 if(y[1]<2) else 1)    
        
            for d in range(num_person):          
                for i in range(n):                        
                    z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                    pairs += [[x[z1], x[z2]]]  #index 0 /  1
                    y_pairs += [[y[z1], y[z2]]]
                    domain_labels += [[if_domain(y[z1]),if_domain(y[z2])]]
                    
                    inc = np.random.randint(1, num_person)
                    dn = (d + inc) % num_person
                    z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                    pairs += [[x[z1], x[z2]]]
                    y_pairs += [[y[z1], y[z2]]]
                    labels += [1, 0]
                    domain_labels += [[if_domain(y[z1]),if_domain(y[z2])]]
                if d == 0 : print('len',len(y_pairs))
   
            return np.array(pairs), np.array(labels), np.array(domain_labels), np.array(y_pairs)

    
         # create training+test positive and negative pairs same lable and angle
        digit_indices = []
        for i in range(self.ratio[0]): #0~72 共73
            if(i==33): continue
        #     for j in range(num_angle):
            digit_indices.append(np.where((y_train[:,0]==i))[0])
        tr_pairs, tr_y, domain_labels,y_pairs = create_pairs(X_train, y_train, digit_indices)
        tr_pairs0 = tr_pairs[:,0,:,:,:]
        tr_pairs1 = tr_pairs[:,1,:,:,:]
        print('tr_pairs,tr_y ',tr_pairs.shape, tr_y.shape,y_pairs.shape, domain_labels.shape)

        return tr_pairs0, tr_pairs1, tr_y, y_pairs, domain_labels
    

    
#     def DataforTestCL(self, X_test, y_test):
# #         bool_target = (y_test[:,1]== 6) & ((y_train[:,2]>=3)*(y_train[:,2]<=4)) #90 & nm
#         bool_CL = (y_test[:,2]>=3)*(y_test[:,2]<=4) #  CL
#         bool_nm90 = (y_test[:,1]== 6) & ((y_test[:,2]>4)*(y_test[:,2]<11)) #90 & nm
#         X_gallery_nm90 = X_test[np.where(bool_nm90)[0]]
#         y_gallery_nm90 = y_test[np.where(bool_nm90)[0]]
#         X_probe_CL = X_test[np.where(bool_CL)[0]]
#         y_probe_CL = y_test[np.where(bool_CL)[0]]
#         unique, counts = np.unique(y_gallery_nm90[:,0], return_counts=True)
#         print(len(unique.tolist()))
#         print("cl= ",dict(zip(unique, counts)))
#         unique, counts = np.unique(y_probe_CL[:,0], return_counts=True)
#         print(len(unique.tolist()))
#         print("nm90= ",dict(zip(unique, counts)))
        
#         return X_gallery_nm90, y_gallery_nm90, X_probe_CL, y_probe_CL

    def DataforTestCL(self, X_test, y_test):
        gallery_X, prob_X, gallery_Y, prob_Y = train_test_split(X_test, y_test, test_size=0.4, shuffle=True)
#         bool_CL = (y_test[:,2]>=3)*(y_test[:,2]<=4) #  CL
        g_bool_nm =  ((gallery_Y[:,2]>4)*(gallery_Y[:,2]<11)) #90 & nm
        p_bool_CL = (prob_Y[:,2]>=3)*(prob_Y[:,2]<=4) #  CL
        X_gallery_nm = gallery_X[np.where(g_bool_nm)[0]]
        y_gallery_nm = gallery_Y[np.where(g_bool_nm)[0]]
        X_probe_CL = prob_X[np.where(p_bool_CL)[0]]
        y_probe_CL = prob_Y[np.where(p_bool_CL)[0]]
        unique, counts = np.unique(y_gallery_nm[:,2], return_counts=True)
        print(len(unique.tolist()))
        print("cl= ",dict(zip(unique, counts)))
        unique, counts = np.unique(y_probe_CL[:,2], return_counts=True)
        print(len(unique.tolist()))
        print("nm90= ",dict(zip(unique, counts)))
        
        return X_gallery_nm, y_gallery_nm, X_probe_CL, y_probe_CL
    
#     def DataforTestCL(self, X_test, y_test): #G nm90 /P nm90
        
#         gallery_X, prob_X, gallery_Y, prob_Y = train_test_split(X_test, y_test, test_size=0.4, shuffle=True)
# #         bool_CL = (y_test[:,2]>=3)*(y_test[:,2]<=4) #  CL
#         g_bool_nm90 = (gallery_Y[:,1]== 6) & ((gallery_Y[:,2]>4)*(gallery_Y[:,2]<11)) #90 & nm
#         p_bool_nm90 = (prob_Y[:,1]== 6) & ((prob_Y[:,2]>4)*(prob_Y[:,2]<11)) #90 & nm
#         X_gallery_nm90 = gallery_X[np.where(g_bool_nm90)[0]]
#         y_gallery_nm90 = gallery_Y[np.where(g_bool_nm90)[0]]
#         X_probe_nm90 = prob_X[np.where(p_bool_nm90)[0]]
#         y_probe_nm90 = prob_Y[np.where(p_bool_nm90)[0]]
#         unique, counts = np.unique(y_gallery_nm90[:,0], return_counts=True)
#         print(len(unique.tolist()))
#         print("nm90= ",dict(zip(unique, counts)))
#         unique, counts = np.unique(y_probe_nm90[:,0], return_counts=True)
#         print(len(unique.tolist()))
#         print("nm90= ",dict(zip(unique, counts)))
        
#         return X_gallery_nm90, y_gallery_nm90, X_probe_nm90, y_probe_nm90
        
#         def DataforPair(self, Xtrain_T, ytrain_T, Xtrain_S, ytrain_S):
#         def create_pairs(Xtrain_T, Xtrain_S, digit_indices_T,digit_indices_S):
#             '''Positive and negative pair creation.
#             Alternates between positive and negative pairs.
#             '''
#             pairs = []
#             labels = []
#             domain_labels = []
#             num_person = len(digit_indices_T)
#             print('num_person', num_person)
#             for d in range(num_person):
#                 numT = len(digit_indices_T[d])
#                 numS = len(digit_indices_S[d])
#                 print("d,numT,numS",d,numT,numS)

#             n_T = min([len(digit_indices_T[d]) for d in range(num_person)] )- 1
#             n_S = min([len(digit_indices_S[d]) for d in range(num_person)] )- 1
#             print("n_T, n_S", n_T, n_S)
#             n = min(n_T,n_S)
#             print("min of T/S: ", n)

#             for d in range(num_person): #62 people
#                 for i in range(100):
#                     inc_T = np.random.randint(1, len(digit_indices_T[d])-1)
#                     inc_S = np.random.randint(1, len(digit_indices_S[d])-1)
#                     z1, z2 = digit_indices_T[d][inc_T], digit_indices_S[d][inc_S]
#                     pairs += [[Xtrain_T[z1], Xtrain_S[z2]]]  #index 0 /  1
                    
#                     inc = np.random.randint(1, num_person)
#                     dn = (d + inc) % num_person
#                     inc_Sdn = np.random.randint(1, len(digit_indices_S[dn])-1)
#                     z1, z2 = digit_indices_T[d][inc_T], digit_indices_S[dn][inc_Sdn]
#                     pairs += [[Xtrain_T[z1], Xtrain_S[z2]]]
#                     labels += [1, 0]     #每次加都會加在裡面
#                     domain_labels += [[1,0],[1,0]]
#             return np.array(pairs), np.array(labels) , np.array(domain_labels)

#         digit_indices_T = []
#         digit_indices_S = []
#         for i in range(62,124): #0~72 共73
#         #     for j in range(num_angle):
#                 digit_indices_T.append(np.where((ytrain_T[:,0]==i))[0])
#                 print(digit_indices_T)
#                 digit_indices_S.append(np.where((ytrain_S[:,0]==i))[0])
#         tr_pairs, tr_y, domain_labels = create_pairs(Xtrain_T, Xtrain_S, digit_indices_T,digit_indices_S)
#         print('tr_pairs,tr_y ',tr_pairs.shape, tr_y.shape,domain_labels.shape)

#         return tr_pairs, tr_y, domain_labels