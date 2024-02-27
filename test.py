import plasma.search_engines as search_engines
import pandas as pd

df = pd.read_feather('icd10.feather')

engine = search_engines.GraphMatcher(df['name'])
engine.run('Hen suyễn')