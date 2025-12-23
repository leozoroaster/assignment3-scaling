import spicy
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "isoflops_curves.json"

with open(DATA_PATH, 'r') as file:
    data = json.load(file)

P=[]
C=[]
L=[]

for data_point in data:
    P.append(data_point['parameters'])
    C.append(data_point['compute_budget'])
    L.append(data_point['final_loss'])

print(P)
