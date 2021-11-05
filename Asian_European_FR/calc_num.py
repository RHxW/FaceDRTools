import os
data_root = r'D:\celeb_deepglint_112\imgs/'
if __name__ == '__main__':

    id_list = os.listdir(data_root)

    f_in = open('./less8_list.txt', 'w')
    for id_name in id_list:

        img_list = os.listdir(os.path.join(data_root, id_name))
        img_num = len(img_list)
        if img_num < 8:
            f_in.writelines([str(id_name), '\t', str(img_num), '\n'])

    f_in.close()
