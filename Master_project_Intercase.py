#!/usr/bin/env python
# coding: utf-8

# In[8]:


##package needed to do the exercise
import pandas as pd #package for reading and using dataset
import numpy as np #package needed for inserting value in dataset


# In[9]:


df=pd.read_csv('C:/Users/s151675/Documents/Jaar_master_2/Kwartiel 3,4/Master project/Data aangepast/data_for_prediction/data_complete.csv', delimiter = ';') #read csv file
df_inter=pd.read_csv('C:/Users/s151675/Documents/Jaar_master_2/Kwartiel 3,4/Master project/data/201026 Aanlevering SEH/201026 Aanlevering SEH/201026 SEH Basis en Triage_NOKIDS_aangepast.csv', delimiter = ';') #read csv file


# In[10]:


df #visualize dataset


# In[11]:


df_inter


# In[ ]:


df_inter.iloc[:,0]
#make a list for every patient, (start and end time, nr, specialism)
list_date_y_r = []
list_date_m_r =[]
list_date_d_r =[]
list_time_r =[]
list_date_y_e = []
list_date_m_e =[]
list_date_d_e =[]
list_time_e =[]
list_spec = []
list_pnr = []
#update this list for every patient
for i in range(0,len(df_inter)):
    pnr_float = df_inter.iloc[i,0]
    pnr_str = str(pnr_float)
    pnr = int(pnr_str[0:len(pnr_str)+2])
    spec = str(df_inter.iloc[i,4])
    reg_date = str(df_inter.iloc[i,7])
    reg_date_y = reg_date[0:4]
    reg_date_m = reg_date[5:7]
    reg_date_d = reg_date[8:11]
    reg_date_time = float(df_inter.iloc[i,8])
    list_date_y_r.append(reg_date_y)
    list_date_m_r.append(reg_date_m)
    list_date_d_r.append(reg_date_d)
    list_time_r.append(reg_date_time)
    
    reg_date = str(df_inter.iloc[i,13])
    reg_date_y = reg_date[0:4]
    reg_date_m = reg_date[5:7]
    reg_date_d = reg_date[8:11]
    reg_date_time = float(df_inter.iloc[i,14])
    list_date_y_e.append(reg_date_y)
    list_date_m_e.append(reg_date_m)
    list_date_d_e.append(reg_date_d)
    list_time_e.append(reg_date_time)
    
    list_pnr.append(pnr)
    list_spec.append(spec)
#     print(reg_date_y)
#     print(reg_date_m)
#     print(reg_date_d)
#     print(reg_date_time)
# print(list_time[0]<=list_time[1])
# print(list_time[0]>=list_time[1])
# print(list_time[0]==list_time[1])
# print(list_time[0]!=list_time[1])
list_ic_same =[] ##list inter-case (how much patient with same specialism in process)
#compare all patients and add one value if another patients is also in the process at the beginning
for patient in range(0,len(df_inter)):
    count = 0
#     print("check patient", patient, "of", len(df_inter))
    for o_patient in range(0,len(df_inter)):
        if(list_spec[patient]==list_spec[o_patient]):
            if(list_date_y_e[o_patient]<list_date_y_r[patient]):#in the future only include patient which end time is not register (yet)
                count += 0
            else:
                if(list_date_m_e[o_patient]<list_date_m_r[patient] and list_date_y_e[o_patient]==list_date_y_r[patient]):
                    count += 0
                else:
                    if(list_date_d_e[o_patient]<list_date_d_r[patient] and list_date_m_e[o_patient]==list_date_m_r[patient] and list_date_y_e[o_patient]==list_date_y_r[patient]):
                        count += 0
                    else:
                        if(list_time_e[o_patient]<list_time_r[patient] and list_date_d_e[o_patient]==list_date_d_r[patient] and list_date_m_e[o_patient]==list_date_m_r[patient] and list_date_y_e[o_patient]==list_date_y_r[patient]):
                            count += 0
                        else:
                            if(list_date_y_r[o_patient]<list_date_y_r[patient]):
                                count += 1
                            else:
                                if(list_date_y_r[o_patient]==list_date_y_r[patient]):
                                    if(list_date_m_r[o_patient]<list_date_m_r[patient]):
                                        count +=1
                                    else:
                                        if(list_date_m_r[o_patient]==list_date_m_r[patient]):
                                            if(list_date_d_r[o_patient]<list_date_d_r[patient]):
                                                count +=1
                                            else:
                                                if(list_date_d_r[o_patient]==list_date_d_r[patient]):
                                                    if(list_time_r[o_patient]<list_time_r[patient]):
                                                        count +=1
    list_ic_same.append(count)
#     print("check patient", patient,"(", list_pnr[patient], ") of", len(df_inter), "value =", list_ic_same[patient])


# In[ ]:


df_inter.iloc[:,0]
#make a list for every patient, (start and end time, nr, specialism)
list_date_y_r = []
list_date_m_r =[]
list_date_d_r =[]
list_time_r =[]
list_date_y_e = []
list_date_m_e =[]
list_date_d_e =[]
list_time_e =[]
list_spec = []
list_pnr = []
#update this list for every patient
for i in range(0,len(df_inter)):
    pnr_float = df_inter.iloc[i,0]
    pnr_str = str(pnr_float)
    pnr = int(pnr_str[0:len(pnr_str)+2])
    spec = str(df_inter.iloc[i,4])
    reg_date = str(df_inter.iloc[i,7])
    reg_date_y = reg_date[0:4]
    reg_date_m = reg_date[5:7]
    reg_date_d = reg_date[8:11]
    reg_date_time = float(df_inter.iloc[i,8])
    list_date_y_r.append(reg_date_y)
    list_date_m_r.append(reg_date_m)
    list_date_d_r.append(reg_date_d)
    list_time_r.append(reg_date_time)
    
    reg_date = str(df_inter.iloc[i,13])
    reg_date_y = reg_date[0:4]
    reg_date_m = reg_date[5:7]
    reg_date_d = reg_date[8:11]
    reg_date_time = float(df_inter.iloc[i,14])
    list_date_y_e.append(reg_date_y)
    list_date_m_e.append(reg_date_m)
    list_date_d_e.append(reg_date_d)
    list_time_e.append(reg_date_time)
    
    list_pnr.append(pnr)
    list_spec.append(spec)
#     print(reg_date_y)
#     print(reg_date_m)
#     print(reg_date_d)
#     print(reg_date_time)
# print(list_time[0]<=list_time[1])
# print(list_time[0]>=list_time[1])
# print(list_time[0]==list_time[1])
# print(list_time[0]!=list_time[1])
list_ic =[] ##list inter-case (how much patient total in process)
#compare all patients and add one value if another patients is also in the process at the beginning
for patient in range(0,len(df_inter)):
    count = 0
#     print("check patient", patient, "of", len(df_inter))
    for o_patient in range(0,len(df_inter)):
        if(list_date_y_e[o_patient]<list_date_y_r[patient]):#in the future only include patient which end time is not register (yet)
            count += 0
        else:
            if(list_date_m_e[o_patient]<list_date_m_r[patient] and list_date_y_e[o_patient]==list_date_y_r[patient]):
                count += 0
            else:
                if(list_date_d_e[o_patient]<list_date_d_r[patient] and list_date_m_e[o_patient]==list_date_m_r[patient] and list_date_y_e[o_patient]==list_date_y_r[patient]):
                    count += 0
                else:
                    if(list_time_e[o_patient]<list_time_r[patient] and list_date_d_e[o_patient]==list_date_d_r[patient] and list_date_m_e[o_patient]==list_date_m_r[patient] and list_date_y_e[o_patient]==list_date_y_r[patient]):
                        count += 0
                    else:
                        if(list_date_y_r[o_patient]<list_date_y_r[patient]):
                            count += 1
                        else:
                            if(list_date_y_r[o_patient]==list_date_y_r[patient]):
                                if(list_date_m_r[o_patient]<list_date_m_r[patient]):
                                    count +=1
                                else:
                                    if(list_date_m_r[o_patient]==list_date_m_r[patient]):
                                        if(list_date_d_r[o_patient]<list_date_d_r[patient]):
                                            count +=1
                                        else:
                                            if(list_date_d_r[o_patient]==list_date_d_r[patient]):
                                                if(list_time_r[o_patient]<list_time_r[patient]):
                                                    count +=1
    list_ic.append(count)
#     print("check patient", patient,"(", list_pnr[patient], ") of", len(df_inter), "value =", list_ic[patient])


# In[ ]:





# In[ ]:


#add extra column for the inter-case features which include the number of patients which are in the process at registering time of corresponding patient
# list_nrpatients_same = np.zeros(len(df))
# list_nrpatients = np.zeros(len(df))
# for i in range(0,len(df)):
#     print(i, "of", len(df))
#     for j in range(i,len(df_inter)):
#         if(df.iloc[i,0]==list_pnr[j]):
#             list_nrpatients_same[i] = list_ic_same[j]
#             list_nrpatients[i] = list_ic[j]
#             break
df_empty = pd.DataFrame(list_pnr)
print(df_empty)
df_empty.insert(1,"nr_patients_same_spec", list_ic_same)
df_empty.insert(2,"nr_patients_total", list_ic)

df_empty.to_csv('C:/Users/s151675/Documents/Jaar_master_2/Kwartiel 3,4/Master project/Data aangepast/Data_for_prediction/data_complete_inter_plustotal.csv')           


# In[ ]:





# In[ ]:




