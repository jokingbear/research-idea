import plasma.search_engines as search_engines
import pandas as pd

df = pd.read_feather('icd10.feather')

engine = search_engines.GraphMatcher(df['name'])
results = engine.run('Viêm dạ dày nhiễm H.pylori')
results['confidence'] = 2 / (1 / results['substring_matching_score'] + 1 / results['word_coverage_score'])