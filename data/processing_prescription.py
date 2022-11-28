'''
select patients with icustay in MIMIC-IV/prescriptions.csv
'''
from numpy.core.records import record
import pandas as pd
from datetime import datetime
import pdb
import dill
import numpy as np
from collections import Counter, defaultdict
from typing import Dict
from rdkit import Chem
from rdkit.Chem import BRICS
from traceback import print_exc
import random


def filter(prescriptions, icustays):
    a = icustays[['subject_id', 'hadm_id']]
    a = a[~a.duplicated()]
    # print(a.subject_id)
    prescriptions_filtered = prescriptions[prescriptions.subject_id.isin(a.subject_id)]
    prescriptions_filtered = prescriptions[prescriptions.hadm_id.isin(a.hadm_id)]
    return prescriptions_filtered

if __name__ == '__main__':
    prescriptions_file = "/data/qychen/open-source/data_ndc_v1/MIMIC-IV/prescriptions.csv"
    icustays_file = "/data/qychen/open-source/data_ndc_v1/MIMIC-IV/icustays.csv"

    prescriptions = pd.read_csv(prescriptions_file, dtype={'ndc': str})
    icustays = pd.read_csv(icustays_file)

    prescriptions_filtered = filter(prescriptions, icustays)

    # print(prescriptions_filtered)
    prescriptions_filtered.to_csv("/data/qychen/open-source/data_ndc_v1/MIMIC-IV/prescriptions_filtered.csv", index=False)

    # print("?????????????????????")
    # prescriptions_filtered = pd.read_csv("/data/qychen/open-source/data_ndc_v1/MIMIC-IV/prescriptions_filtered.csv")
    # prescriptions = pd.read_csv("/data/qychen/open-source/data_ndc_v1/MIMIC-IV/prescriptions.csv")
    # print(prescriptions_filtered)
    # print("-----------")
    # print(prescriptions)
    # prescriptions_filtered.to_csv("/data/qychen/open-source/data_ndc_v1/MIMIC-IV/prescriptions_filtered.csv", index=False)


