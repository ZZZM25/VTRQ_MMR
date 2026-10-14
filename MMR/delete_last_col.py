from pathlib import Path
INPUT_FILE=Path(r'E:\MMR_Trajectory_range\xian_edges.txt')
OUTPUT_FILE=Path('xian_edges_no_last_col.txt')

def remove_last_column(input_file:Path,output_file:Path)->None:
    with input_file.open('r',encoding='utf-8') as fin, output_file.open('w',encoding='utf-8',newline='') as fout:
        count=0
        for line in fin:
            s=line.strip()
            if not s: fout.write('\n'); continue
            cols=s.split(); fout.write(' '.join(cols[:-1] if len(cols)>=2 else cols)+'\n'); count+=1
    print('Processing completed'); print('Input file:',input_file); print('Output file:',output_file); print('Number of processed lines:',count)

if __name__=='__main__': remove_last_column(INPUT_FILE,OUTPUT_FILE)
