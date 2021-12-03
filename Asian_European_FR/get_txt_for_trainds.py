import os

def get_txt_for_train_dataset(txt_file, get_id_num, txt_save_path, ori_root, dst_root):
    with open(txt_file) as f:
        lns = f.readlines()
    if ori_root[-1] != '/':
        ori_root += '/'
    if dst_root[-1] != '/':
        dst_root += '/'
    new_lns = []
    for ln in lns[:get_id_num]:
        if ln[-1] == '\n':
            ln = ln[:-1]
        new_ln = ori_root + ln + ' ' + dst_root + ln + '\n'
        new_lns.append(new_ln)
    
    with open(txt_save_path, 'w') as f:
        f.writelines(new_lns)
    
    
if __name__ == '__main__':
    txt_file = '/home/songhui/Projects/Asian_European_FR/european_idlist.txt'
    get_id_num = 90000
    txt_save_path = '/home/traindata/facedata/european_90k_new_1.txt'
    ori_root = '/home/traindata/facedata/train_glint360_96/imgs/'
    dst_root = '/home/traindata/facedata/train_data_new_1/imgs/'
    get_txt_for_train_dataset(txt_file, get_id_num, txt_save_path, ori_root, dst_root)