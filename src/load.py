#!/share/pkg.8/python3/3.12.4/install/bin/python3

import pandas as pd
import numpy as np
import datetime
import pickle
import re
from sklearn.preprocessing import LabelEncoder


TRAINING_DATA = "/projectnb/peaclab-mon/boztop/module_files/data/accounting.2024"

class LoadFilterData:
    def __init__(self, file_path=TRAINING_DATA):
        self.file_path = file_path
        self.col_headers = ["qname", "hostname", "group", "owner", "job_name", "job_number", "account", "priority", 
             "submission_time", "start_time", "end_time", "failed", "exit_status", "ru_wallclock", 
             "ru_utime", "ru_stime", "ru_maxrss", "ru_ixrss", "ru_ismrss", "ru_idrss", "ru_isrss", 
             "ru_minflt", "ru_majflt", "ru_nswap", "ru_inblock", "ru_oublock", "ru_msgsnd", 
             "ru_msgrcv", "ru_nsignals", "ru_nvcsw", "ru_nivcsw", "project", "department", "granted_pe", 
             "slots", "task_number", "cpu", "mem", "io", "category", "iow", "pe_taskid", "maxvmem", 
             "arid", "ar_submission_time"] # make this shorter! use only needed!!
        self.use_col_headers = ["group", "owner", "job_name",  "submission_time", "start_time", "end_time", "failed", "exit_status", "ru_wallclock", 
             "ru_utime", "ru_stime", "ru_maxrss", "ru_isrss", "project", "granted_pe", 
             "slots", "cpu", "mem", "io", "category"]     
        self.train_features = ['group', 'owner', 'job_name',  'granted_pe','slots','h_rt']  
        self.df = None

    def load_and_filter_data(self):
        """
        This method loads data from a CSV file and applies initial filtering.
        """
        self.df = pd.read_csv(self.file_path, encoding='utf-8', skiprows=4, names=self.col_headers, sep=':')
        self.df = self.df[self.df['ru_wallclock'] != 0]
        self.df = self.df[self.df['failed'] == 0]
        self.df = self.df[self.df['exit_status'] == 0]
        self.df['lag_time'] = self.df['start_time'] - self.df['submission_time']
        self.df = self.df[self.df['lag_time'] > 0]
        self.df = self.df.dropna()

    def calculate_additional_features(self):
        """
        This method calculates additional features based on existing columns.
        """
        self.df['execution_time'] = self.df['end_time'] - self.df['start_time']
        self.df['total_cpu_time'] = self.df['ru_utime'] + self.df['ru_stime']
        self.df['ncpu'] = np.ceil(self.df['total_cpu_time'] / self.df['ru_wallclock'])
        self.df['cpu_waste'] = self.df['slots'] - self.df['ncpu']
        self.df['submission_time'] = self.df['submission_time'].apply(
            lambda x: datetime.datetime.fromtimestamp(x)
        )

    def drop_unused_columns(self):
        """
        This method drops columns that are not required for the analysis.
        """
        self.df = self.df.drop(['account', 'job_number', 'failed', 'exit_status', 'iow', 
                                'pe_taskid', 'arid', 'ar_submission_time'], axis=1)

    def parse_h_rt_flag(self, column):
        """
        This method parses the h_rt (hard runtime) flag from the category column.
        """
        parsed_data = []
        for entry in column:
            matches = re.findall(r'-l ([^\s-]+)', entry)
            h_rt_value = None
            for match in matches:
                for item in match.split(','):
                    if item.startswith('h_rt='):
                        _, value = item.split('=', 1)
                        h_rt_value = int(value)
            parsed_data.append({'h_rt': h_rt_value})
        return parsed_data

    def add_h_rt_column(self):
        """
        This method parses the h_rt flag from the category column and adds a new column to the DataFrame.
        """
        parsed_flags = self.parse_h_rt_flag(self.df['category'])
        parsed_df = pd.DataFrame(parsed_flags)
        self.df = pd.concat([self.df, parsed_df], axis=1)
        self.df['h_rt'] = self.df['h_rt'].fillna(12 * 60 * 60).astype(int)

    def encode_categorical_features(self, categorical_features):
        """
        This method encodes categorical features using LabelEncoder.
        """
        label_encoders = {col: LabelEncoder() for col in categorical_features}
        for col in categorical_features:
            self.df[col] = label_encoders[col].fit_transform(self.df[col])
        
        self.label_encoders = label_encoders

    def preprocess_data(self):
        """
        This method preprocesses the data by loading, filtering, and encoding features.
        """
        print("Loading and filtering data...\n")
        self.load_and_filter_data()
        
        print("Calculating additional features...\n")
        self.calculate_additional_features()
        
        print("Dropping unused columns...\n")
        self.drop_unused_columns()
        
        print("Parsing h_rt flags...\n")
        self.add_h_rt_column()
        print("Parsing done!\n")
        
        print("Encoding categorical features...\n")
        categorical_submission_features = ['group', 'owner', 'job_name', 'department', 'granted_pe']
        self.encode_categorical_features(categorical_submission_features)
        print("Encoding done!\n")

        print("Saving label encoders...\n")
        with open('/projectnb/peaclab-mon/boztop/module_files/models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("Label encoders saved!\n")
        
        return self.df
