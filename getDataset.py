#%% Imports
import sys                                                                       
import os
os.environ['MNE_USE_NUMBA'] = 'false'                                            #避免使用numba speedup
import mne
import numpy as np
from IPython.display import clear_output

#Import of self created python script
sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')                              # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
import settings
from tools import test_edf_corrupted_info, get_date_edf                           ## (edf_corrupted, edf_info) false没坏，edf_metadata; edf measurment date yyyy-mm-dd


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
            patient_id = filename[9:17]                                                  # aaaaaaaa 病人ID
            session_id = filename[19:22]                                                 # s001 病人会话次数
            take_id = filename[24:27]                                                    # t000 第一个转换而来的token
            path_to_edf = os.path.join(dirpath, filename)                                # os.path.join 合并路径和文件名= 文件完整路径
            print(path_to_edf)
            #print(import_progress)
            import_progress += 1
            if import_progress%700==0:   #this loop displays the progress  循环显示进度  除以700余0==每700次给用户汇报一次进度
                clear_output(wait=True)  # 清除前面的进度
                print("Importing dataset:"+str(import_progress/700) + "%") 
                
            corrupted, edf_info = test_edf_corrupted_info(path_to_edf)                    # false, metadata
            if not corrupted:
                if patient_id in patients:                               # 添加到patient字典 如果有就是说先前已经有这个病人id的档案了，添加在这个Key下面
                    patients[patient_id].append((str(edf_info['meas_date'])[0:10], 's_' + session_id + '_t_' + take_id, path_to_edf, edf_info))
                else:                                                    # 新病人 ，新建病例
                    patients[patient_id] = [(str(edf_info['meas_date'])[0:10], 's_' + session_id + '_t_' + take_id, path_to_edf, edf_info)]
            
    total_numbers_dataset(patients)        

    return patients

def total_numbers_dataset(patients):                   # 打印出关于这个数据集patients[]的一些统计信息:病人，token；sessions
    #print information about dataset
    print('Number of patients:', len(patients.keys()))   #病人数
    eeg_total = 0
    sessions_total = 0
    
    for patient_id in patients.keys():                 # 一共会诊几次;
        sessions = []
        for (session_id, _ , _, _) in patients[patient_id]:
            if session_id not in sessions:
                sessions.append(session_id)
            
        
        sessions_total += len(sessions)
        eeg_total += len(patients[patient_id])
    
    print('Number of EEGs', eeg_total)
    print('Number of Sessions', sessions_total)
    
    return(len(patients.keys()),eeg_total, sessions_total)
 
def rand_bool(probability_true):  #probability values is an float in [0.0, 1.0)
    n = np.random.random()
    if n <= probability_true:
        return True
    else:
        return False
    
def seperate_session_patient(patient, session_id):
    patient_session = []
    patient_without_session = []
    
    for take in patient:
        session_id_take, _, _, _ = take
        if session_id_take == session_id:
            patient_session.append(take)
        else:
            patient_without_session.append(take)
    #if len(patient) != len(patient_session) + len(patient_without_session):
    #    print('Error while sperating sessions')
    return patient_session, patient_without_session
                     
    
def split_train_val_test(dataset_dict):  
    
    #splited datasets created
    train_dataset = {}
    validation_dataset = {}
    test_dataset = {} 
    
    patients = dataset_dict.keys()
    for pat in patients:
        # deal with patients with 1 session and 1 take in this session
        if len(dataset_dict[pat]) == 1:
            #probability of 80% to get seletced for train dataset
            train_bool = rand_bool(0.8)
            if train_bool:
                train_dataset[pat] = dataset_dict[pat]
            else:
                validation_bool = rand_bool(0.5)
                if validation_bool:
                    validation_dataset[pat] = dataset_dict[pat]
                else:
                    test_dataset[pat] = dataset_dict[pat]
        #deal with patients with more than 1 take
        else:
            # get all sessions from this patient, and a second list with all the takes in this session
            sessions = []  # list with session ids
            sessions_takes = [] # list with same order as sessions, at index of an session id there is an list with all takes in this session
            for edf in dataset_dict[pat]:
                ses_id, take_id, _, _ = edf
                if ses_id not in sessions:
                    sessions.append(ses_id)
                    sessions_takes.append([take_id])
                else:
                    session_index = sessions.index(ses_id)
                    sessions_takes[session_index].append(take_id)   
                    
            # deal with patients with 1 session
            if len(sessions) == 1:
                # choose 10% of this patients as test/Validation
                train_bool = rand_bool(0.9)
                if train_bool:
                    train_bool = rand_bool(0.4)
                    #choose 40% of all patients as train, and from 60% choose one take to validation/test
                    if train_bool: # add to train
                        train_dataset[pat] = dataset_dict[pat]
                    
                    else:  #choose the take and add to test/validation
                        number_takes = len(sessions_takes[0])
                        val_test_take = np.random.randint(0, number_takes)  #random int between 0 (inclusive) and number of all takes (exclusive) --> we get the indexes of the take list
                        # now we add this take to test/validation (50/50)
                        # --> this patients are used for in session accuracy
                        val_test = [dataset_dict[pat][val_test_take]]
                                                
                        validation_bool = rand_bool(0.5)
                        if validation_bool:
                            validation_dataset[pat] = val_test
                        else:
                            test_dataset[pat] = val_test
                        
                        #add the rest of the takes to train
                        train_takes = []
                        for take in dataset_dict[pat]:
                            if take not in val_test:
                                train_takes.append(take)
                        
                        train_dataset[pat] = train_takes
                        
                # --> the 10% get added to validation/test
                else:
                    validation_bool = rand_bool(0.5)
                    if validation_bool:
                        validation_dataset[pat] = dataset_dict[pat]
                    else:
                        test_dataset[pat] = dataset_dict[pat]
                        
            #deal with patients with more than 1 sessions
            else:
                number_sessions = len(sessions) #number of sessions from an patient
                #with a probability of 10% chose 1 session(with all takes) from this patient
                test_val_bool = rand_bool(0.1)
                if test_val_bool:
                    number_session = np.random.randint(0, number_sessions)
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
                    #this helps keeping the train data high enough in comparison to the approach above which chooses completly random
                    test_val_bool = rand_bool(0.2)
                    if test_val_bool:
                        least_takes = [sessions[0],len(sessions_takes[0])]  # (session_id, number of takes)
                        for i in range (1, len(sessions)):
                            if len(sessions_takes[i]) < least_takes[1]:
                                #choose the session with the least entry takes
                                least_takes = [sessions[i],len(sessions_takes[i])]
                            elif len(sessions_takes[i]) == least_takes[1]:
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
                
    return (train_dataset, validation_dataset, test_dataset)

def convert_to_pandas_dataframe(dataset_dict):
    convert_list = []
    keys = dataset_dict.keys()
    for key in keys:
        patient_id = str(key)
        takes = dataset_dict[key] 
        for take in takes:
            session_id, take_id, path_to_edf, info_meta = take
            convert_list.append([patient_id, session_id, take_id, path_to_edf, info_meta])
    df = pd.DataFrame(np.array(convert_list), columns=['patient_id', 'session_id/date', 'take_id', 'path_to_edf', 'edf_info'])
    
    return df





# %%
