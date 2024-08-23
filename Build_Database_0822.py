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

# Test EDF file and get metainformation
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
        edf_sfreq = f.info['sfreq']   
        del f
    except:                                                     ## error
        print('Import error on file: ' +  path_to_edf)
        edf_corrupted = True
    
    
    # Open edf as plain text to get attributes from header
    with open(path_to_edf, 'rb') as h:
       header = h.read(123).decode('utf-8')
    # Modify header to get attributes
    modified_header = header.split(' ')
    sex_str = modified_header[8]
    if sex_str in ['F', 'f']:
     sex ='f'
    elif sex_str in ['M', 'm']:
     sex = 'm'
    else:
     sex = 'x'
            
    age_str = modified_header[11][4:]
    if age_str.isdigit():
     age = int(age_str)
    else: # missing/wrong value
     age = -1    
    
    return (edf_corrupted, edf_info, edf_time, edf_ch_names, edf_sfreq,sex, age)


#%% Definitions

def get_subjects(path):                                                                    ## 遍历所有文件寻找. edf文件，并转化为dictionary
    subjects = {} #key: subject_id ; value: list [tuple (date_of_the_take, sesssion_id+take_id, path_to_edf, meta_file), (), ()]   
    
    
    #walk through all files and get all edf files in the given directory                   os.walk 遍历目录树 三元组（dirpath, dirnames, filenames） 
    #these paths are added to the subjects dict with (and metadata dict if needed)         dirpath是一个字符串，表示当前正在遍历的目录的路径；
    # dirpath    是一个string 字符串，表示当前正在遍历的目录的路径；
    # dirnames   是一个list   列表，包含dirpath下所有子目录的名字。注意，这个列表不会包含子目录下进一步的子目录的名字。                                                                                
    # filenames  是一个list   列表，包含dirpath下所有非目录文件的名字。就是文件名


    import_progress = 0  #initializing the progress displaying
    
    for dirpath, dirnames, filenames in os.walk(path):
        #walk recursive from a certain path to find all files
        
        for filename in [f for f in filenames if f.endswith(".edf")]: #filter the files to the ones ending with edf   筛选文件 .edf  f=有几个文件 00000000_aaaaaaaa_s001_t000.edf
            #get all information from the file name
            subject_id = filename[0:8]                                                  # aaaaaaaa 病人ID 9-16 不包括17
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
    
                
            corrupted, edf_info, edf_time, edf_chan, edf_sfreq, sex, age = test_edf_corrupted_info(path_to_edf)                    # false, metadata, time
            channels_set = set(edf_chan)
            
            
            if (required_channels_1.issubset(channels_set) or required_channels_2.issubset(channels_set)):
             if not corrupted:
                if subject_id in subjects:                               # 添加到subject字典 如果有就是说先前已经有这个病人id的档案了，添加在这个Key下面
                    subjects[subject_id].append(('s_' + session_id, 't_' + take_id, sex, age, str(edf_time) ,str(edf_info['meas_date'])[0:10],path_to_edf, edf_chan,edf_sfreq,edf_info ))
                else:                                                    # 新病人 ，新建病例
                    subjects[subject_id] = [('s_' + session_id, 't_' + take_id,sex, age,str(edf_time) ,str(edf_info['meas_date'])[0:10],path_to_edf, edf_chan,edf_sfreq, edf_info)]
            
    total_numbers_dataset(subjects)        

    return subjects

def total_numbers_dataset(subjects):                   # 打印出关于这个数据集subjects[]的一些统计总信息：病人数，会诊数量，EEG一共几段
    #print information about dataset
    print('Number of subjects:', len(subjects.keys()))   #病人数
    eeg_total = 0
    sessions_total = 0
    #b=0
    for subject_id in subjects.keys():                 # 一共会诊几次     subject_id='aaaaaaxx'
        sessions = []
        for (session_id, _ ,_,_ ,_, _, _,_, _,_ ) in subjects[subject_id]:     # 只考虑这个subjects[aaaaaaxx]中的第一位日期，用session_id代指, 即S_001_2002-01-01，也就是这一天做过几次Session（其实日期可能是某一年，不是具体到某一天）
            if session_id not in sessions:
                sessions.append(session_id)
                #print(sessions)
        
        sessions_total += len(sessions)
        eeg_total += len(subjects[subject_id])
    
    print('Number of EEGs', eeg_total)
    print('Number of Sessions', sessions_total)
   
    return(len(subjects.keys()),eeg_total, sessions_total)

def rand_bool(probability_true):                   #probability values is an float in [0.0, 1.0)  probability_true 
    n = np.random.random()                                  #
    if n <= probability_true:                               #
        return True                     
    else:
        return False                  

def seperate_session_subject(subject, session_id):             #通过S_00X_2002-01-01为基准 将原来的病人dic分为两部分  这里的subject就是''aaaaaaax','这个人病人的所有data,Session,take等
    subject_session = []
    subject_without_session = []
    
    for P_X in subject:                                       # s_001
        session_id_1 , _ ,_ ,_,_ , _, _, _,_,_ = P_X
        if session_id_1 == session_id:
            subject_session.append(P_X)
        else:
            subject_without_session.append(P_X)
    #if len(subject) != len(subject_session) + len(subject_without_session):
    #    print('Error while sperating sessions')
    return subject_session, subject_without_session 

def get_dataset(subject_dic):
    # filter datasets based on user-defined configurations; 
    # configurations: 1. subjects have at least 2 sessions; 2. edf times at least 10mins~ 600sec
    
    
    S_1_T_1 = 0            
    S_1_T_n = 0
    S_n_T_1 = 0            
    S_n_T_n = 0
    
    Dataset_dic = {}
    subjects_id = subject_dic.keys()
    for p_X in subjects_id:               ##遍历每一个subject
        sessions = []  # list with session ids                                 ## subject's session id like S001 S002 S003
                                                                ##  ['s_001',             's_002',            's_003']          
        sessions_takes = [] # list with same order as sessions, at index of an session id there is an list with all takes in this session  类似矩阵 列是2002-02-02等 行是s_001_t_000,s_001_t_001,...
        
                                                                    #[['t_000', 't_001'], ['t_000', 't_001'], ['t_000', 't_001', 't_002']]
        session_dates = set()
        
        Add_Dataset = []
        
        for Pat_X in subject_dic[p_X]:                                 # 相当于 p_X 这个病人的所有.edf,一个一个循环[_,_,_,_,_,_,_]
                s_id, token_id, _ ,_,test_time, sessions_date,_, _,_ ,_= Pat_X
                # filter out the edf_time less than 10 mins
                if int(test_time)> 600:
                
                     Add_Dataset.append(Pat_X) 
                     session_dates.add(sessions_date)
                     #print('ADD=',Add_Dataset)
                     if s_id not in sessions:                                         ## S001下没有其他token了，session[]新建下一个id S002；session_takes[]直接加上
                      sessions.append(s_id)                                        
                      sessions_takes.append([token_id])  
                     else:                                                              ## S001下还有 t002,t003...找到S001的index位置，插入take的id
                      session_index = sessions.index(s_id)
                      sessions_takes[session_index].append(token_id)
        # 1. deal :filter out the subjects with 1 session and 1 take in this session            
        if len(Add_Dataset) == 1:
           S_1_T_1+=1
           Add_Dataset[0] = Add_Dataset[0] + ('A',)  # Add the 'A' label to the only element in Add_Dataset
           Dataset_dic[p_X] = Add_Dataset
           continue
        # 2. deal with subjects with 1 session
        if len(sessions) == 1:                                                  
            S_1_T_n+=1
            for i in range(len(Add_Dataset)):
              Add_Dataset[i] = Add_Dataset[i] + ('B',)    # value 'B' means:Subject has 1 Session but more Tokens/Recordings
        # 3. deal with subjects with more than 1 sessions
        else:
             if len(session_dates) == 1:
              S_n_T_1+=1  
              for i in range(len(Add_Dataset)):
                 Add_Dataset[i] = Add_Dataset[i] + ('C',) # value 'C' means:Subject has 2 or more Sessions but in same date
             else:  
              S_n_T_n+=1     
              for i in range(len(Add_Dataset)):  
                 Add_Dataset[i] = Add_Dataset[i] + ('D',) # value 'D' means:Subject has 2 or more Sessions and in diff dates
                                                        # value 'D' means: sessions for re-identification / target session
        Dataset_dic[p_X] = Add_Dataset
    
    
    # Add additional inforamtion for filter from 'Dataset_dic'
    # ADD re-identification session number; reidx_session_id
    for p_X in Dataset_dic.keys():
        sessions = Dataset_dic[p_X]
        sorted_sessions = sorted(sessions, key=lambda x: x[0])
        session_id_dic = {}
        current_id = 1
        prev_session_date = None

        for i in range(len(sorted_sessions)):
         session_id = sorted_sessions[i][0]
         session_date = sorted_sessions[i][5]
         # Sessions with the same date!! are set to the same number
         if session_id not in session_id_dic:
             if prev_session_date is not None and session_date == prev_session_date:
                session_id_dic[session_id] = session_id_dic[prev_session_id]
             else:
                session_id_dic[session_id] = current_id
                current_id += 1

         sorted_sessions[i] = sorted_sessions[i] + (session_id_dic[session_id],)
         prev_session_id = session_id
         prev_session_date = session_date

         Dataset_dic[p_X] = sorted_sessions
        
    # Identify the last sessions of each subject and add '1'
    for p_X in Dataset_dic.keys():
        sessions_search = Dataset_dic[p_X]
    # Find the largest/last reidx_session_id
        if not sessions_search:            # Check if sessions[] are empty
            continue
        max_session_id = max(sessions_search, key=lambda x: x[-1])[-1]
        for i in range(len(sessions_search)):
         if sessions_search[i][-1] == max_session_id:
                sessions_search[i] = sessions_search[i] + (1,)
         else:
                sessions_search[i] = sessions_search[i] + (0,)
        
        Dataset_dic[p_X] = sessions_search
    
    # Find the first recording of each 'reidx_session_id=1'
    for p_X in Dataset_dic.keys():
     sessions_search = Dataset_dic[p_X]
     found_first = False
     for i in range(len(sessions_search)):
        # Check if the first sessions of subject (reidx_session_id=tenth column) = '1'
        if sessions_search[i][11] == 1:
            if not found_first:
                # Set the new column to 1 for the first match/ first  recording as training input!
                sessions_search[i] = sessions_search[i] + (1,)
                found_first = True
            else:
                # Set the new column to 0 for all other matches
                sessions_search[i] = sessions_search[i] + (0,)
        else:
            # Set the new column to 0 for all non-matching rows
            sessions_search[i] = sessions_search[i] + (0,)

     Dataset_dic[p_X] = sessions_search
                
    print('S_1_T_1=',S_1_T_1)
    print('S_1_T_n=',S_1_T_n)
    print('S_n_T_1=',S_n_T_1)
    print('S_n_T_n=',S_n_T_n)
    return Dataset_dic
 
 
# Dic to dataframe        
def convert_to_pandas_dataframe(dataset_dict):
    # subjects_dataframe
    convert_list = []
    keys = dataset_dict.keys()                                       # P_id=subject id 'aaaaaaac'
    for P_id in keys:
        subject_id = str(P_id)                                       # P_record
        P_record = dataset_dict[P_id] 
        for take in P_record:
            session_id, take_id, sex, age, session_time,session_date,path_to_edf, channel_edf,edf_sfreq, info_meta, session_number,reidx_session_id,last_s,first_t = take
            convert_list.append([subject_id, session_id, take_id, sex, age, session_time,session_date,path_to_edf, channel_edf, edf_sfreq,info_meta, session_number, reidx_session_id,last_s,first_t])
    df = pd.DataFrame(np.array(convert_list), columns=['subject_id', 'session_id', 'token_id','sex', 'age ','edf_time','session_date','path_to_edf','edf_channel', 'edf_sample_freq','edf_info','subject_class','reidx_session_id','last_session','first_recording'])
    
    return df             

# split dataset from session #2 to val_set and test_set
def split_val_test_with_probability(df, val_probability=0.2):
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for _, row in df.iterrows():
        if rand_bool(val_probability):
            val_df = pd.concat([val_df, row.to_frame().T], ignore_index=True)
        else:
            test_df = pd.concat([test_df, row.to_frame().T], ignore_index=True)

    return val_df, test_df


def get_Reidentifiable_subsets(dataframe_df):
    # Filter by 'first_recording' = 1 and with 2 sessions
    filtered_df = dataframe_df[(dataframe_df['first_recording'] == 1) & (dataframe_df['subject_class'] == 'D')]
    
    # Check the amount of recording
    total_rows = len(filtered_df)
    if total_rows == 0:
        raise ValueError("No recordings found for the Reidentifiable subset.")
    
    return filtered_df


# creat dataset to val_test_set
def get_val_test_subsets(dataframe_df):
    
    val_test_df = pd.DataFrame()
    vt_df = dataframe_df[(dataframe_df['reidx_session_id'] == 2) & (dataframe_df['subject_class'] == 'D')]
    # filter by 'first_recording' = 1 
    subject_list = vt_df['subject_id'].unique()

    # 遍历每个 subject_id 并进一步筛选数据
    for subject in subject_list:
        # 从已经筛选好的 val_test_df 中再次抽取每个 subject 的一行数据
        subject_data = vt_df[vt_df['subject_id'] == subject]
        if not subject_data.empty:
            # 从筛选出的数据中随机选择一行，加入到 val_test_df
            selected_row = subject_data.sample(n=1)
            val_test_df = pd.concat([val_test_df, selected_row], ignore_index=True)
    
    return val_test_df
    

# 创建 challenge_set
def create_challenge_set(dataframe, challenge_size=10):
    # 过滤条件：'subject_class' == 'D' 且 'last_session' == 1
    filtered_df = dataframe[(dataframe['subject_class'] == 'D') & (dataframe['last_session'] == 1)]
    
    # 检查可用行数
    unique_subjects = filtered_df['subject_id'].unique()
    if len(unique_subjects) < challenge_size:
        raise ValueError("Not enough unique subjects for the held-out set.")
    
    # 随机选择 10 个不同的 subject_id
    selected_subjects = np.random.choice(unique_subjects, challenge_size, replace=False)
    
    # 从每个 selected_subject 中随机选择一行
    challenge_set = pd.DataFrame()
    for subject in selected_subjects:
        subject_rows = filtered_df[filtered_df['subject_id'] == subject]
        selected_row = subject_rows.sample(n=1)  # 随机选择一行
        challenge_set = challenge_set.append(selected_row, ignore_index=True)
    
    return challenge_set






#%%


#path_to_edf_files = 'E:\\EEG data\\edf\\'
path_to_edf_files = 'C:\\Users\\49152\\Desktop\\MA\\Code\\000\\'
subjects_data = get_subjects(path_to_edf_files)
raw_dataset_number = total_numbers_dataset(subjects_data)
subjects_dataset = get_dataset(subjects_data)
subjects_dataframe = convert_to_pandas_dataframe(subjects_dataset)




# export subset to Excel
def export_subset_to_excel(df, filename):
    df.to_excel(filename, index=False)


# defination path   定义文件路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
Reidentifiable_subset_path = os.path.join(desktop_path, "Reidentifiable_subset.xlsx")
dataframe_path = os.path.join(desktop_path, "dataframe.xlsx")
val_test_subset_path = os.path.join(desktop_path, "val_test_subset.xlsx")
val_subset_path = os.path.join(desktop_path, "val_subset.xlsx")
test_subset_path = os.path.join(desktop_path, "test_subset.xlsx")
#challenge_set_path = os.path.join(desktop_path, "challenge_set_set.xlsx")

# export dataframe to desltop  
#export_subset_to_excel(subset_dataframe, Reidentifiable_subset_path)
export_subset_to_excel(subjects_dataframe, dataframe_path)
dataframe_df = pd.read_excel(dataframe_path)


# get subset, trainset, validation subset, test subset
subset_dataframe = get_Reidentifiable_subsets(dataframe_df)
val_test_subset_dataframe = get_val_test_subsets(dataframe_df)
#challenge_set = create_challenge_set(dataframe_df)
val_set, test_set = split_val_test_with_probability(val_test_subset_dataframe, val_probability=0.2)


#export dataset
export_subset_to_excel(subset_dataframe, Reidentifiable_subset_path)
export_subset_to_excel(val_test_subset_dataframe, val_test_subset_path)
#export_subset_to_excel(challenge_set, challenge_set_path)
export_subset_to_excel(val_set, val_subset_path)
export_subset_to_excel(test_set, test_subset_path)
