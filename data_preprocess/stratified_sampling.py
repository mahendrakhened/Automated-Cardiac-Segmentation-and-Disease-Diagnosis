import numpy as np
import os, sys, shutil
# print sys.path
# sys.path.append("..") 
import errno
np.random.seed(42)


# Refer:
# http://www.echopedia.org/wiki/Left_Ventricular_Dimensions
# https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
# https://en.wikipedia.org/wiki/Body_surface_area
# 30 normal subjects - NOR
NORMAL = 'NOR'
# 30 patients with previous myocardial infarction 
# (ejection fraction of the left ventricle lower than 40% and several myocardial segments with abnormal contraction) - MINF
MINF = 'MINF'
# 30 patients with dilated cardiomyopathy 
# (diastolic left ventricular volume >100 mL/m2 and an ejection fraction of the left ventricle lower than 40%) - DCM
DCM = 'DCM'
# 30 patients with hypertrophic cardiomyopathy 
# (left ventricular cardiac mass high than 110 g/m2,
# several myocardial segments with a thickness higher than 15 mm in diastole and a normal ejecetion fraction) - HCM
HCM = 'HCM'
# 30 patients with abnormal right ventricle (volume of the right ventricular 
# cavity higher than 110 mL/m2 or ejection fraction of the rigth ventricle lower than 40%) - RV
RV = 'RV'
def copy(src, dest):
  """
  Copy function
  """
  try:
      shutil.copytree(src, dest, ignore=shutil.ignore_patterns())
  except OSError as e:
      # If the error was caused because the source wasn't a directory
      if e.errno == errno.ENOTDIR:
          shutil.copy(src, dest)
      else:
          print('Directory not copied. Error: %s' % e)

def read_patient_cfg(path):
  """
  Reads patient data in the cfg file and returns a dictionary
  """
  patient_info = {}
  with open(os.path.join(path, 'Info.cfg')) as f_in:
    for line in f_in:
      l = line.rstrip().split(": ")
      patient_info[l[0]] = l[1]
  return patient_info
     
def group_patient_cases(src_path, out_path, force=False):
  """ Group the patient data according to cardiac pathology""" 

  cases = sorted(next(os.walk(src_path))[1])
  dest_path = os.path.join(out_path, 'Patient_Groups')
  if force:
    shutil.rmtree(dest_path)
  if os.path.exists(dest_path):
    return dest_path  

  os.makedirs(dest_path)
  os.mkdir(os.path.join(dest_path, NORMAL))
  os.mkdir(os.path.join(dest_path, MINF))
  os.mkdir(os.path.join(dest_path, DCM))
  os.mkdir(os.path.join(dest_path, HCM))
  os.mkdir(os.path.join(dest_path, RV))

  for case in cases:
    full_path = os.path.join(src_path, case)
    copy(full_path, os.path.join(dest_path,\
        read_patient_cfg(full_path)['Group'], case))

def generate_train_validate_test_set(src_path, dest_path):
  """
  Split the data into 70:15:15 for train-validate-test set
  arg: path: input data path
  """
  SPLIT_TRAIN = 0.7
  SPLIT_VALID = 0.15

  dest_path = os.path.join(dest_path,'dataset')
  if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
  os.makedirs(os.path.join(dest_path, 'train_set'))  
  os.makedirs(os.path.join(dest_path, 'validation_set'))  
  os.makedirs(os.path.join(dest_path, 'test_set'))  
  # print (src_path)
  groups = next(os.walk(src_path))[1]
  for group in groups:
    group_path = next(os.walk(os.path.join(src_path, group)))[0]
    patient_folders = next(os.walk(group_path))[1]
    np.random.shuffle(patient_folders)
    train_ = patient_folders[0:int(SPLIT_TRAIN*len(patient_folders))]
    valid_ = patient_folders[int(SPLIT_TRAIN*len(patient_folders)): 
                 int((SPLIT_TRAIN+SPLIT_VALID)*len(patient_folders))]
    test_ = patient_folders[int((SPLIT_TRAIN+SPLIT_VALID)*len(patient_folders)):]
    for patient in train_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'train_set', patient))

    for patient in valid_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'validation_set', patient))

    for patient in test_:
      folder_path = os.path.join(group_path, patient)
      copy(folder_path, os.path.join(dest_path, 'test_set', patient))


if __name__ == '__main__':
  # Path to ACDC training database
  complete_data_path = '../../ACDC_DataSet/training'
  dest_path = '../../processed_acdc_dataset'
  group_path = '../../processed_acdc_dataset/Patient_Groups'
  group_patient_cases(complete_data_path, dest_path)
  generate_train_validate_test_set(group_path, dest_path)