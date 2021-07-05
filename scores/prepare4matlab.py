import os
from tqdm import tqdm

def convert2txt(txt_file):
    
    filename = txt_file.split('/')[-1].split('.txt')[0]
    output_tar = os.path.join('./', filename + '_tar.txt')
    output_non = os.path.join('./', filename + '_non.txt')
    
    with open(output_tar, 'w') as wt:
        with open(txt_file) as f:
            lines = f.readlines()

            for line in tqdm(lines):
                score = line.split(' ')[1]
                decision = line.split(' ')[2]
                
                if decision == 'bonafide\n':
                    wt.write('%s\n'%(score))
                    
    with open(output_non, 'w') as wn:
        with open(txt_file) as f:
            lines = f.readlines()

            for line in tqdm(lines):
                score = line.split(' ')[1]
                decision = line.split(' ')[2]
                
                if decision == 'spoof\n':
                    wn.write('%s\n'%(score))

if __name__ == "__main__":
    
    file = './lfcc_ecapa512cfst_ocs_19dev_score.txt'
    convert2txt(file)
