# SEQ-RF
Sequence of Random Forest 

This repository consist of 5 different files for prediction and 2 files for pre processing:

# Preprocessing:

Master_project_preprocess_timestamp = Transform two columns, one for date and one for time, into one column

Master_project_Intercase = Make two different intercase features, one for the number of patients in the hospital when a patient arrived and one for number of patients with same specialism when a patient arrived

# Prediction: 

Master_project_random_forest_baseline = Random Forest for predicting overstaying with the use of patient data

Master_project_random_forest_baseline + Intercase = Random Forest for predicting overstaying with the use of patient data and Intercase features

Master_project_random_forest_buildup-consult-opname-NOIC = Sequence of Random Forests for predicting overstayinh with the use of patient data and activities

Master_project_random_forest_buildup-inter-consult-opname = Sequence of Random Forests for predicting overstayinh with the use of patient data, activities and Intercase features

Master_project_random_forest_buildup-performed_activities = Random Forest for predicting overstaying with the use of patient data, Intercase features and performed activities.
