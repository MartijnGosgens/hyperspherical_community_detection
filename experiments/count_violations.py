import pandas as pd
from os import listdir,path

def print_violations():
    violations = 0
    total=0
    relative_errors = []
    output_dir = path.join(path.dirname(__file__), 'output')
    for csv in listdir(output_dir):
        df = pd.read_csv(path.join(output_dir,csv))
        total+=len(df)
        df_violations = df[(df[r'$d_a(q,b(C))$']/df[r'$d_a(q,b(T))$']-1)>0]
        relative_errors+=list(df_violations[r'$d_a(q,b(C))$']/df_violations[r'$d_a(q,b(T))$']-1)
        violations+=len(df_violations)
    print('#instances where d_a(q,b(C))>d_a(q,b(T)):',violations,'/',total)
    print('#instances where d_a(q,b(C))>1.01*d_a(q,b(T)):',len([r for r in relative_errors if r>0.01]),'/',total)
    print('#instances where d_a(q,b(C))>1.02*d_a(q,b(T)):',len([r for r in relative_errors if r>0.02]),'/',total)