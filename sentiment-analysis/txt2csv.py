
import csv

headers = ['evaluation', 'label']

rows = []

with open("train.txt", "r", encoding="utf-8-sig") as f:    #打开文件
    for line in f.readlines():
        # data = f.read()   #读取文件
        # print(line)
        # 1 positive 2 neg 3 netural

        # print(line.split(" ")[1])
        # print(line.split(" ")[0])
        rows.append([line.split(" ")[1], line.split(" ")[0]])

print(rows)

with open('train.csv','w', encoding="utf-8")as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)