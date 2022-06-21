import os

fw = open('test_data.txt', 'w', encoding = 'utf-8')
dirpath = '../../data/test'
for filename in os.listdir(dirpath):
    fw.write(os.path.join(dirpath, filename) + '\n')
fw.close()
fw = open('dev_data.txt', 'w', encoding = 'utf-8')
dirpath = '../../data/dev/images'
for filename in os.listdir(dirpath):
    fw.write(os.path.join(dirpath, filename) + '\n')
fw.close()
print('done')
