import pandas as pd
from glob import glob
import sys
import os
import pprint

pp = pprint.PrettyPrinter(indent=4)
suffix = '' if len(sys.argv) < 2 else sys.argv[1] 
files = glob(os.path.join(suffix, '*.csv'))
files = sorted(files)

record = []
for f in files:
    df = pd.read_csv(f)
    mean_score = df['scores'].mean()
    record.append((f, mean_score))
    print('File %s\t has average score %.5f' % (f, mean_score))

print('\n')
sorted_record = sorted(record, key=lambda t: t[1], reverse=True)
for f, s in sorted_record:
    print('File %s\t has average score %.5f' % (f, s))
