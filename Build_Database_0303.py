#%% Imports
import sys                                                                       
import os
os.environ['MNE_USE_NUMBA'] = 'false'                                            #避免使用numba speedup
import mne
import numpy as np
from IPython.display import clear_output
import pandas as pd
#Import of self created python script
sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')                              # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
import settings
from tools import test_edf_corrupted_info, get_date_edf                           ## (edf_corrupted, edf_info) false没坏，edf_metadata; edf measurment date yyyy-mm-dd


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#setup Frameworks
mne.set_log_level('WARNING')

#%% Definitions

def get_patients(path):                                                                    ## 遍历所有文件寻找. edf文件，并转化为dictionary
    patients = {} #key: patient_id ; value: list [tuple (date_of_the_take, sesssion_id+take_id, path_to_edf, meta_file), (), ()]   
    
    
    #walk through all files and get all edf files in the given directory                   os.walk 遍历目录树 三元组（dirpath, dirnames, filenames） 
    #these paths are added to the patients dict with (and metadata dict if needed)         dirpath是一个字符串，表示当前正在遍历的目录的路径；
    # dirpath    是一个string 字符串，表示当前正在遍历的目录的路径；
    # dirnames   是一个list   列表，包含dirpath下所有子目录的名字。注意，这个列表不会包含子目录下进一步的子目录的名字。                                                                                
    # filenames  是一个list   列表，包含dirpath下所有非目录文件的名字。就是文件名


    import_progress = 0  #initializing the progress displaying
    
    for dirpath, dirnames, filenames in os.walk(path):
        #walk recursive from a certain path to find all files
        
        for filename in [f for f in filenames if f.endswith(".edf")]: #filter the files to the ones ending with edf   筛选文件 .edf  f=有几个文件 00000000_aaaaaaaa_s001_t000.edf
            #get all information from the file name
            patient_id = filename[9:17]                                                  # aaaaaaaa 病人ID 9-16 不包括17
            session_id = filename[19:22]                                                 # s001 病人会话次数
            take_id = filename[24:27]                                                    # t000 第一个转换而来的token
            path_to_edf = os.path.join(dirpath, filename)                                # os.path.join 合并路径和文件名= 文件完整路径
            #print(path_to_edf)
            import_progress += 1
            if import_progress%700==0:   #this loop displays the progress  循环显示进度  除以700余0==每700次给用户汇报一次进度
                clear_output(wait=True)  # 清除前面的进度
                print("Importing dataset:"+str(import_progress/700) + "%") 
                
            corrupted, edf_info = test_edf_corrupted_info(path_to_edf)                    # false, metadata
            if not corrupted:
                if patient_id in patients:                               # 添加到patient字典 如果有就是说先前已经有这个病人id的档案了，添加在这个Key下面
                    patients[patient_id].append(('s_' + session_id+'_'+str(edf_info['meas_date'])[0:10], 't_' + take_id, path_to_edf, edf_info))
                else:                                                    # 新病人 ，新建病例
                    patients[patient_id] = [('s_' + session_id+'_'+str(edf_info['meas_date'])[0:10], 't_' + take_id, path_to_edf, edf_info)]
            
    total_numbers_dataset(patients)        

    return patients

def total_numbers_dataset(patients):                   # 打印出关于这个数据集patients[]的一些统计总信息：病人数，会诊数量，EEG一共几段
    #print information about dataset
    print('Number of patients:', len(patients.keys()))   #病人数
    eeg_total = 0
    sessions_total = 0
    #b=0
    for patient_id in patients.keys():                 # 一共会诊几次     patient_id='aaaaaaxx'
        sessions = []
        for (session_id, _ , _, _) in patients[patient_id]:     # 只考虑这个patients[aaaaaaxx]中的第一位日期，用session_id代指, 即S_001_2002-01-01，也就是这一天做过几次Session（其实日期可能是某一年，不是具体到某一天）
            if session_id not in sessions:
                sessions.append(session_id)
                #print(sessions)
        
        sessions_total += len(sessions)
        eeg_total += len(patients[patient_id])
    
    print('Number of EEGs', eeg_total)
    print('Number of Sessions', sessions_total)
   
    return(len(patients.keys()),eeg_total, sessions_total)

def rand_bool(probability_true):                   #probability values is an float in [0.0, 1.0)  表示返回True的概率 
    n = np.random.random()                                  #通过调整probability_true来控制返回True的概率
    if n <= probability_true:                               #n小于或等于probability_true，则函数返回True
        return True                     
    else:
        return False                  ##这个函数可以在需要根据某个概率来决定事件是否发生的情况下使用，例如模拟实验结果、决策树分支等。

def seperate_session_patient(patient, session_id):             #通过S_00X_2002-01-01为基准 将原来的病人dic分为两部分  这里的patient就是''aaaaaaax','这个人病人的所有data,Session,take等
    patient_session = []
    patient_without_session = []
    
    for P_X in patient:                                       # s_001_2022-02-01
        session_id_data,_, _, _ = P_X
        if session_id_data == session_id:
            patient_session.append(P_X)
        else:
            patient_without_session.append(P_X)
    #if len(patient) != len(patient_session) + len(patient_without_session):
    #    print('Error while sperating sessions')
    return patient_session, patient_without_session 

def get_dataset(patient_dic):
    #filter datasets based on user-defined configurations;
    S_1_T_1 = 0            
    S_1_T_n = 0
    patients_id = patient_dic.keys()
    for pat_X in patients_id:
        # 1. deal :filter out the patients with 1 session and 1 take in this session            
        if len(patient_dic[pat_X]) == 1:
           S_1_T_1+=1  
    
    

 return dataset_dic

def split_train_val_test(dataset_dict):                                           ## Patients[] --> 训练集（train_dataset）、验证集（validation_dataset）和测试集（test_dataset）
    
    #splited datasets created                                                     ## 初始化
    train_dataset = {}                                                            
    validation_dataset = {}
    test_dataset = {} 
    
    patients = dataset_dict.keys()                                                ## 读取Patients[]dic： patients="aaaaaaac" ；  dataset_dict.keys()=所有’aaaaaaax‘  
    for pat in patients:
        # 1. deal :filter out the patients with 1 session and 1 take in this session            
        if len(dataset_dict[pat]) == 1:                                           ## 选出只有一段session 一段 token 的病人PAT "aaaaaaxx"
                  
        else:
            # get all sessions from this patient, and a second list with all the takes in this session  
            sessions = []  # list with session ids                                 ## "PAT"这个病人的所有 session id like S001 S002 S003
            sessions_takes = [] # list with same order as sessions, at index of an session id there is an list with all takes in this session  类似矩阵 列是2002-02-02等 行是s_001_t_000,s_001_t_001,...
            for Pat_X in dataset_dict[pat]:
                s_id, token_id, _, _ = Pat_X                  # 读取这两个值 s_id 是S_00X_2022-01-01 ,token_id是't_00X'等
                if s_id not in sessions:                                         ## S001下没有其他token了，session[]新建下一个id S002；session_takes[]直接加上
                    sessions.append(s_id)                                        
                    sessions_takes.append([token_id])                               
                else:                                                              ## S001下还有 t002,t003...找到S001的index位置，插入take的id
                    session_index = sessions.index(s_id)
                    sessions_takes[session_index].append(token_id)   
                    
            #2. deal with patients with 1 session
            if len(sessions) == 1:                                                  ## 病人'PAT_a'只有一个Session 
                S_1_T_n+=1
                '''# choose 10% of this patients as test/Validation
                train_bool = rand_bool(0.9)                                         ## 分配 90%到训练集 ； 
                if train_bool:
                    train_bool = rand_bool(0.4)                                     ## 40% 这90%的直接进训练集
                    #choose 40% of all patients as train, and from 60% choose one take to validation/test
                    if train_bool: # add to train                   
                        train_dataset[pat] = dataset_dict[pat]
                    
                    else:  #choose the take and add to test/validation
                        number_tokens = len(sessions_takes[0])                        ## 这个病人唯一的Session有几个Token 
                        val_test_take = np.random.randint(0, number_tokens)  #random int between 0 (inclusive) and number of all takes (exclusive) --> we get the indexes of the take list 随机选一个token的index
                        # now we add this take to test/validation (50/50)
                        # --> this patients are used for in session accuracy
                        val_test = [dataset_dict[pat][val_test_take]]
                                                
                        validation_bool = rand_bool(0.5)                              ## 50% 分配入验证或测试
                        if validation_bool:
                            validation_dataset[pat] = val_test
                        else:
                            test_dataset[pat] = val_test
                        
                        #add the rest of the takes to train                           ##除了随机的token以外的进入训练集
                        train_takes = []
                        for take in dataset_dict[pat]:
                            if take not in val_test:
                                train_takes.append(take)
                        
                        train_dataset[pat] = train_takes
                        
                # --> the 10% get added to validation/test                             ## 剩余一开始10%没被选上训练集的 50/50 V or Test
                else:
                    validation_bool = rand_bool(0.5)
                    if validation_bool:
                        validation_dataset[pat] = dataset_dict[pat]
                    else:
                        test_dataset[pat] = dataset_dict[pat]'''
                        
            #3. deal with patients with more than 1 sessions
            else:
                number_sessions = len(sessions) #number of sessions from an patient          ## 这个PAT_X具体有几个 Session n=？
                #with a probability of 10% chose 1 session(with all takes) from this patient    10%
                test_val_bool = rand_bool(0.1)     
                if test_val_bool:
                    number_session = np.random.randint(0, number_sessions)                      ##随机选一个Session
                    choosen_session_id = sessions[number_session] #chose the sesion randomly
                    
                    takes_from_choosen_session, takes_without_choosen_session = seperate_session_patient(dataset_dict[pat], choosen_session_id)
                    #add all takes not from the choosen session to train
                    train_dataset[pat] = takes_without_choosen_session
                    #add the choosen session to validation/test (50/50)
                    validation_bool = rand_bool(0.5)
                    if validation_bool:
                        validation_dataset[pat] = takes_from_choosen_session
                    else:
                        test_dataset[pat] = takes_from_choosen_session
                    
                else:
                    #with a probability of 20% (only if no sessios has been choosen so far from this patient) take the session with the LEAST takes as test/validation
                    #this helps keeping the train data high enough in comparison to the approach above which chooses completly random 和完全随机相比 保证训练集的质量
                    test_val_bool = rand_bool(0.2)         
                    if test_val_bool:
                        least_takes = [sessions[0],len(sessions_takes[0])]  # (session_id, number of takes)
                        for i in range (1, len(sessions)):                            ## 循环for 找最小token的Session
                            if len(sessions_takes[i]) < least_takes[1]:
                                #choose the session with the least entry takes
                                least_takes = [sessions[i],len(sessions_takes[i])]
                            elif len(sessions_takes[i]) == least_takes[1]:             ## 如果有两个最小值的Session 50\50随便选一个 
                                #if the session has the same number of takes as the current minimum session, randomly choose (50/50) on of the 2 sessions 
                                choose = rand_bool(0.5)
                                if choose:
                                    least_takes = [sessions[i],len(sessions_takes[i])]
                        
                        #choose the session with the least entries as test/val session            
                        choosen_session_id = least_takes[0]
                        
                        takes_from_choosen_session, takes_without_choosen_session = seperate_session_patient(dataset_dict[pat], choosen_session_id)
                        #add all takes not from the choosen session to train
                        train_dataset[pat] = takes_without_choosen_session
                        #add the choosen session 
                        validation_bool = rand_bool(0.5)
                        if validation_bool:
                            validation_dataset[pat] = takes_from_choosen_session
                        else:
                            test_dataset[pat] = takes_from_choosen_session
                              
                    else:
                        train_dataset[pat] = dataset_dict[pat]
        
    print('S_1_T_1=',S_1_T_1)
    print('S_1_T_n=',S_1_T_n)
                
    return (train_dataset, validation_dataset, test_dataset)


def convert_to_pandas_dataframe(dataset_dict):
    convert_list = []
    keys = dataset_dict.keys()                                       # P_id=Patient id 'aaaaaaac'
    for P_id in keys:
        patient_id = str(P_id)                                       # P_record
        P_record = dataset_dict[P_id] 
        for take in P_record:
            session_id, take_id, path_to_edf, info_meta = take
            convert_list.append([patient_id, session_id, take_id, path_to_edf, info_meta])
    df = pd.DataFrame(np.array(convert_list), columns=['patient_id', 'session_id_date', 'token_id', 'path_to_edf', 'edf_info'])
    
    return df





# %%

path_to_edf_files = 'C:\\Users\\49152\\Desktop\\MA\\Code'


patients_data = get_patients(path_to_edf_files)
patients_dataset = total_numbers_dataset(patients_data)
train_dataset, validation_dataset, test_dataset=split_train_val_test(patients_data)   
#print(patients_data['aaaaaaab'])
#print(patient_session)
#print(patient_without_session)
#print(patients_dataset)

def export_dataset_to_txt(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, values in dataset.items():
            file.write(f"Patient ID: {key}\n")
            for value in values:
                file.write(f"{value}\n")
            file.write("\n")                              # 换行

# defination path   定义文件路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
train_file_path = os.path.join(desktop_path, "train_dataset.txt")
validation_file_path = os.path.join(desktop_path, "validation_dataset.txt")
test_file_path = os.path.join(desktop_path, "test_dataset.txt")

# export dataset to desltop  导出数据集
export_dataset_to_txt(train_dataset, train_file_path)
export_dataset_to_txt(validation_dataset, validation_file_path)
export_dataset_to_txt(test_dataset, test_file_path)


train_dataset_pd=convert_to_pandas_dataframe(train_dataset)
test_dataset_pd=convert_to_pandas_dataframe(test_dataset)
validation_dataset_pd=convert_to_pandas_dataframe(validation_dataset)