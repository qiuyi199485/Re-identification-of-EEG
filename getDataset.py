#%% Imports
import sys
import os
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
import numpy as np
from IPython.display import clear_output

#Import of self created python script
sys.path.insert(1, 'C:\Users\49152\Desktop\MA\Code')
import settings
from tools import test_edf_corrupted_info, get_date_edf


#setup Frameworks
mne.set_log_level('WARNING')

#%% Definitions

def get_patients(path):
    patients = {} #key: patient_id ; value: list [tuple (date_of_the_take, sesssion_id+take_id, path_to_edf, meta_file), (), ()]   
    
    
    #walk through all files and get all edf files in the given directory
    #these paths are added to the patients dict with (and metadata dict if needed)
    
    import_progress = 0  #initializing the progress displaying
    
    
    for dirpath, dirnames, filenames in os.walk(path):
        #walk recursive from a certain path to find all files
        
        for filename in [f for f in filenames if f.endswith(".edf")]: #filter the files to the ones ending with edf
            #get all information from the file name
            patient_id = filename[0:8]
            session_id = filename[10:13]
            take_id = filename[15:18]
            path_to_edf = os.path.join(dirpath, filename)
            
            import_progress += 1
            if import_progress%700==0:   #this loop displays the progress
                clear_output(wait=True)
                print("Importing dataset:"+str(import_progress/700) + "%") 
                
            corrupted, edf_info = test_edf_corrupted_info(path_to_edf)
            if not corrupted:
                if patient_id in patients:
                    patients[patient_id].append((str(edf_info['meas_date'])[0:10], 's_' + session_id + '_t_' + take_id, path_to_edf, edf_info))
                else:
                    patients[patient_id] = [(str(edf_info['meas_date'])[0:10], 's_' + session_id + '_t_' + take_id, path_to_edf, edf_info)]
    total_numbers_dataset(patients)        

    return patients

def total_numbers_dataset(patients):
    #print information about dataset
    print('Number of patients:', len(patients.keys()))
    eeg_total = 0
    sessions_total = 0
    
    for patient_id in patients.keys():
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




