import pandas as pd
from pathlib import Path
result=pd.read_excel(Path('C:/Users/abhis/Desktop/Fake_Therapist.xlsx'))
result.to_csv('Fake_Therapist1.csv',index=False, sep='$',header=True)