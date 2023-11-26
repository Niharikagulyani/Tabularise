import os 
import pandas as pd

elements_df = pd.read_csv('/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Elements.csv')

simple_tables_indices = elements_df.index[(elements_df['Rows_Count']<=5) & (elements_df['Columns_Count']<=5) ]
print(len(simple_tables_indices))
simple_tables = elements_df['Unnamed: 0'][simple_tables_indices].values.tolist()
simple_tables_file = open("/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Simple_Tables.txt",'w')
for file in simple_tables:
    simple_tables_file.write(file)
    simple_tables_file.write("\n")
simple_tables_file.close()



medium_tables_indices= elements_df.index[((elements_df['Rows_Count']<=15) & (elements_df['Columns_Count']<=5)) |((elements_df['Rows_Count']<=5) & (elements_df['Columns_Count']<=15)) ]
print(len(medium_tables_indices))
medium_tables = elements_df['Unnamed: 0'][medium_tables_indices].values.tolist()
medium_tables_file = open("/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Medium_Tables.txt",'w')
for file in medium_tables:
    medium_tables_file.write(file)
    medium_tables_file.write("\n")
medium_tables_file.close()


complex_tables_indices = elements_df.index[((elements_df['Rows_Count']>15) & (elements_df['Rows_Count']<25) & (elements_df['Columns_Count']<25)  ) | ((elements_df['Rows_Count']<25) & (elements_df['Columns_Count']>15) & (elements_df['Columns_Count']<25)) ]
print(len(complex_tables_indices))
complex_tables = elements_df['Unnamed: 0'][complex_tables_indices].values.tolist()
complex_tables_file = open("/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Complex_Tables.txt",'w')
for file in complex_tables:
    complex_tables_file.write(file)
    complex_tables_file.write("\n")
complex_tables_file.close()

