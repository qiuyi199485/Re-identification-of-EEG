#%% Imports
import sys                                                                       
import os
os.environ['MNE_USE_NUMBA'] = 'false'                                            #避免使用numba speedup
import mne
import numpy as np
from IPython.display import clear_output
import pandas as pd
#Import of self created python script
sys.path.insert( 1,'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')                              # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
#sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')       
import settings
#from tools import test_edf_corrupted_info, get_date_edf                           ## (edf_corrupted, edf_info) false没坏，edf_metadata; edf measurment date yyyy-mm-dd
from tools import get_date_edf 

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#setup Frameworks
mne.set_log_level('WARNING')


def test_edf_corrupted_info(path_to_edf):                      ##check if xx.edf can be read
    edf_corrupted = False
    edf_info = False
    try:
        f = mne.io.read_raw_edf(                                ## mne.io.f = mne对象， 包含了edf的元数据信息metadata
            path_to_edf, 
            preload=False,                                      
            verbose=0, 
            stim_channel=None)
        edf_info = f.info 
        edf_time = int(f.times[-1]) 
        edf_ch_names = f.info['ch_names']## edf可以用mne读出来就是 edf_info=edf所有metadat   
        del f
    except:                                                     ## error
        print('Import error on file: ' +  path_to_edf)
        edf_corrupted = True
        
    return (edf_corrupted, edf_info, edf_time, edf_ch_names)


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
            patient_id = filename[0:8]                                                  # aaaaaaaa 病人ID 9-16 不包括17
            session_id = filename[10:13]                                                 # s001 病人会话次数
            take_id = filename[15:18]                                                    # t000 第一个转换而来的token
            path_to_edf = os.path.join(dirpath, filename)                                # os.path.join 合并路径和文件名= 文件完整路径
            #print(path_to_edf)
            import_progress += 1
            if import_progress%700==0:   #this loop displays the progress  循环显示进度  除以700余0==每700次给用户汇报一次进度
                clear_output(wait=True)  # 清除前面的进度
                print("Importing dataset:"+str(import_progress/700) + "%") 
            
            required_channels_1 = set([
                 "EEG FP1-LE", "EEG FP2-LE", "EEG F7-LE", "EEG F3-LE", "EEG FZ-LE", "EEG F4-LE", "EEG F8-LE", 
                         "EEG A1-LE", "EEG T3-LE", "EEG C3-LE", "EEG CZ-LE", "EEG C4-LE", "EEG T4-LE", "EEG A2-LE", 
                         "EEG T5-LE", "EEG P3-LE", "EEG PZ-LE", "EEG P4-LE", "EEG T6-LE", "EEG O1-LE", "EEG O2-LE"
             ])
            
            required_channels_2 = set([
                 "EEG FP1-REF", "EEG FP2-REF", "EEG F7-REF", "EEG F3-REF", "EEG FZ-REF", "EEG F4-REF", "EEG F8-REF", 
                         "EEG A1-REF", "EEG T3-REF", "EEG C3-REF", "EEG CZ-REF", "EEG C4-REF", "EEG T4-REF", "EEG A2-REF", 
                         "EEG T5-REF", "EEG P3-REF", "EEG PZ-REF", "EEG P4-REF", "EEG T6-REF", "EEG O1-REF", "EEG O2-REF"
             ])
    
                
            corrupted, edf_info, edf_time, edf_chan = test_edf_corrupted_info(path_to_edf)                    # false, metadata, time
            channels_set = set(edf_chan)
            
            
            if (required_channels_1.issubset(channels_set) or required_channels_2.issubset(channels_set)):
             if not corrupted:
                if patient_id in patients:                               # 添加到patient字典 如果有就是说先前已经有这个病人id的档案了，添加在这个Key下面
                    patients[patient_id].append(('s_' + session_id, 't_' + take_id, str(edf_time) ,str(edf_info['meas_date'])[0:10],path_to_edf, edf_chan,edf_info ))
                else:                                                    # 新病人 ，新建病例
                    patients[patient_id] = [('s_' + session_id, 't_' + take_id,str(edf_time) ,str(edf_info['meas_date'])[0:10],path_to_edf, edf_chan, edf_info)]
            
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
        for (session_id, _ ,_, _, _, _,_ ) in patients[patient_id]:     # 只考虑这个patients[aaaaaaxx]中的第一位日期，用session_id代指, 即S_001_2002-01-01，也就是这一天做过几次Session（其实日期可能是某一年，不是具体到某一天）
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
    
    for P_X in patient:                                       # s_001
        session_id_1 , _ ,_ , _, _, _,_ = P_X
        if session_id_1 == session_id:
            patient_session.append(P_X)
        else:
            patient_without_session.append(P_X)
    #if len(patient) != len(patient_session) + len(patient_without_session):
    #    print('Error while sperating sessions')
    return patient_session, patient_without_session 

def get_dataset(patient_dic):
    # filter datasets based on user-defined configurations; 
    # configurations: 1. patients have at least 2 sessions; 2. edf times at least 10mins~ 600sec
    
    
    S_1_T_1 = 0            
    S_1_T_n = 0
    Dataset_dic = {}
    patients_id = patient_dic.keys()
    for p_X in patients_id:
        sessions = []  # list with session ids                                 ## "PAT"这个病人的所有 session id like S001 S002 S003
                                                                ##  ['s_001',             's_002',            's_003']          
        sessions_takes = [] # list with same order as sessions, at index of an session id there is an list with all takes in this session  类似矩阵 列是2002-02-02等 行是s_001_t_000,s_001_t_001,...
                                                                    #[['t_000', 't_001'], ['t_000', 't_001'], ['t_000', 't_001', 't_002']]
        Add_Dataset = []
        
        for Pat_X in patient_dic[p_X]:                                 # 相当于 p_X 这个病人的所有.edf,一个一个循环[_,_,_,_,_,_,_]
                s_id, token_id, test_time, _,_, _ ,_= Pat_X
                # filter out the edf_time less than 10 mins
                if int(test_time)> 600:
                    Add_Dataset.append(Pat_X) 
                    #print('ADD=',Add_Dataset)
                    if s_id not in sessions:                                         ## S001下没有其他token了，session[]新建下一个id S002；session_takes[]直接加上
                     sessions.append(s_id)                                        
                     sessions_takes.append([token_id])  
                    else:                                                              ## S001下还有 t002,t003...找到S001的index位置，插入take的id
                     session_index = sessions.index(s_id)
                     sessions_takes[session_index].append(token_id)
        # 1. deal :filter out the patients with 1 session and 1 take in this session            
        if len(Add_Dataset) == 1:
           S_1_T_1+=1  
        # 2. deal with patients with 1 session
        elif len(sessions) == 1:                                                  
           S_1_T_n+=1
        # 3. deal with patients with more than 1 sessions
        else:
 
         Dataset_dic[p_X] = Add_Dataset
                 
    #print('S_1_T_1=',S_1_T_1)
    #print('S_1_T_n=',S_1_T_n)
    
    return Dataset_dic
        
def convert_to_pandas_dataframe(dataset_dict):
    # patients_dataframe
    convert_list = []
    keys = dataset_dict.keys()                                       # P_id=Patient id 'aaaaaaac'
    for P_id in keys:
        patient_id = str(P_id)                                       # P_record
        P_record = dataset_dict[P_id] 
        for take in P_record:
            session_id, take_id, session_time,session_date,path_to_edf, channel_edf, info_meta = take
            convert_list.append([patient_id, session_id, take_id, session_time,session_date,path_to_edf, channel_edf, info_meta])
    df = pd.DataFrame(np.array(convert_list), columns=['patient_id', 'session_id', 'token_id', 'edf_time','session_date','path_to_edf','edf_channel' ,'edf_info'])
    
    return df             


# split dataset to train_val_test
def split_train_val_test(df):
    train_df = df.sample(frac=0.8, random_state=1)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=1)
    test_df = temp_df.drop(val_df.index)
    
    #print(f"Training set size: {len(train_df)}")
    #print(f"Validation set size: {len(val_df)}")
    #print(f"Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df

def get_challenges_subsets(dataframe_df, subset_size=50, number_subsets=1):
    total_rows = len(dataframe_df)
    if total_rows < subset_size:
        raise ValueError("challenges_subsets bigger than DataFrame")
    subsets = dataframe_df.sample(n=subset_size)
    return subsets

# %%

path_to_edf_files = 'E:\\2024.8.11\\$$$文件夹13545635841\\'
patients_data = get_patients(path_to_edf_files)
raw_dataset_number = total_numbers_dataset(patients_data)
patients_dataset = get_dataset(patients_data)
patients_dataframe = convert_to_pandas_dataframe(patients_dataset)




# export subset to Excel
def export_subset_to_excel(df, filename):
    df.to_excel(filename, index=False)

# export subset to txt
def export_subset_to_txt(df, filename):
    df.to_csv(filename, sep='\t', index=False)


# defination path   定义文件路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")
dataframe_path = os.path.join(desktop_path, "dataframe.xlsx")
train_subset_path = os.path.join(desktop_path, "train_subset.txt")
validation_subset_path = os.path.join(desktop_path, "validation_subset.txt")
test_subset_path = os.path.join(desktop_path, "test_subset.txt")


# export dataset to desltop  导出数据集
#export_subset_to_excel(subset_dataframe, challenges_subset_path)
export_subset_to_excel(patients_dataframe, dataframe_path)
dataframe_df = pd.read_excel(dataframe_path)


# get subset, trainset, validation subset, test subset
subset_dataframe = get_challenges_subsets(dataframe_df)
train_subset, validation_subset, test_subset = split_train_val_test(subset_dataframe)
export_subset_to_excel(subset_dataframe, challenges_subset_path)
export_subset_to_txt(train_subset, train_subset_path)
export_subset_to_txt(validation_subset, validation_subset_path)
export_subset_to_txt(test_subset, test_subset_path)
