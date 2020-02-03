import csv
import random
import math
import matplotlib.pyplot as plot

data_x = []
data_y = []
#membaca file dari bentuk csv
def baca_file(data_file):
    isi_file = []
    with open(data_file) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # print (row)
            x = [float(row[0]), float(row[1])] #memasukkan csv dalam bentuk array multidimensi
            isi_file.append(x)
            data_x.append(float(row[0])) #menyimpan data pada kolom pertama
            data_y.append(float(row[1])) #menyimpada data pada kolom kedua
    return isi_file

#melakukan random cluster
def nilai_random_cluster():
    random_score = []
    #melakukan random nilai sebanyak 15 karena diketahui dari dataset terdapat 15 cluster
    for i in range(15):
        ran_s1 = random.uniform(3,16) #melakukan random nilai yang disesuaikan dengan range nilai dari dataset
        ran_s2 = random.uniform(3,16) #melakukan random nilai yang disesuaikan dengan range nilai dari dataset
        tot_s = [ran_s1, ran_s2]
        # print(tot_s)
        random_score.append(tot_s)
    return random_score

#mengembalikan angka indeks terkecil dari sebuah data
def cari_minimum(a):
    return a.index(min(a))

#menghitung jarak cluster dan data lalu dicari indeks minimum dari hasil jarak yang terkecil
def distance(cluster, data):
    simpan_distance = []
    for i in range(0,15):
        distance = math.sqrt(((data[0] - cluster[i][0]) ** 2) + ((data[1] - cluster[i][1])**2))
        simpan_distance.append(distance)
        x = cari_minimum(simpan_distance)
    return x

#mengupdate nilai weight apabila diketahui nilai tersebut kecil dengan memberikan indeks dari nilai terkecil tersebut di dalam cluster
def nilai_update(learning_rate, isi_min, cluster, data):
    a = cluster[isi_min][0] + (learning_rate * (data[0] - cluster[isi_min][0]))
    b = cluster[isi_min][1] + (learning_rate * (data[1] - cluster[isi_min][1]))
    # masukkan_list.append([a,b])
    return [a,b]

# mengubah nilai sekitar dari nilai terkecil di neighbour rumus dan algoritma ini menyesuaikan dari http://www.ai-junkie.com/ann/som/som3.html
def neighbour_score(jum_iterasi, iterasi_sekarang, isi_min, learning_rate_dirubah, data_random_dataset, cluster):
    #mencari nilai maksimum di dataset yang dari x dan y
    nilai_radius = max(max(data_x), max(data_y)) / 2
    # time constant (jumlah iterasi/lock nilai logaritma maksimum)
    nilai_contant = jum_iterasi / math.log(nilai_radius)
    # mencari neighbour radius = nilai terbesar dengan eksponen -iterasi keberapa saat itu/ time constant
    nilai_tetangga = nilai_radius * math.exp(-(iterasi_sekarang / nilai_contant))
    # perulangan sebanyak 15 kali
    for i in range(0, 15):
        jarak = math.sqrt(((cluster[isi_min][0] - cluster[i][0]) ** 2) + ((cluster[isi_min][1] - cluster[i][1])**2))
        # menghitung euclidian diantara cluster yang didapatkan di pertama kali - dengan semua cluster yang lain
        if jarak < nilai_tetangga:
            nilai_tetha = math.exp(-((jarak) ** 2) / (2 * ((nilai_tetangga) ** 2)))
            cluster[i][0] = cluster[i][0] + nilai_tetha * learning_rate_dirubah * (data_random_dataset[0] - cluster[i][0])
            cluster[i][1] = cluster[i][1] + nilai_tetha * learning_rate_dirubah * (data_random_dataset[1] - cluster[i][1])
    return cluster

#klasifikasi yang digunakan yakni menggunakan dictionary agar key merupakan clusternya dan value nilai yang dimiliki oleh cluster
def klasifikasi(jum_cluster, data, cluster):
    kamus  = {}
    for i in range(0, jum_cluster):
        kamus[i] = []
    for j in range(0, len(data)):
        indeks = distance(cluster, data[j])
        kamus[indeks].append(data[j])
    return kamus

if __name__ == '__main__':
    data = baca_file('Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv')
    # print(data)
    learning_rate = 0.6
    learning_rate_dirubah = 0.6
    jum_cluster = 15
    jum_iterasi = 10000
    cluster = []
    cluster = nilai_random_cluster()
    print(cluster)
    for i in range(0, jum_iterasi):
        data_random_dataset = random.choice(data) #step 1 yakni mengambil nilai random dari dataset
        isi_min = distance(cluster, data_random_dataset) #step 2 mencari nilai weight dari nilai random tersebut dan mengambil nilai indeks terkecil
        cluster[isi_min] = nilai_update(learning_rate, isi_min, cluster, data_random_dataset) #step 3 indeks terkecil yang berisi nilai terkecil akan diupdate
        cluster = neighbour_score(jum_iterasi, i, isi_min, learning_rate_dirubah, data_random_dataset, cluster) #step 4 mengubah nilai tetangganya dengan asumsikan nilai learning_rate akan terus berkurang menyesuaikan dengan jumlah iterasi
        learning_rate_dirubah = learning_rate * math.exp(-(i/jum_iterasi)) #learning rate dirubah dikarenakan untuk mengupdate nilai neighbour
        simpanan = klasifikasi(jum_cluster, data, cluster) #simpan menampung hasil klasifikasi yang berbentuk dictionary
    #melakukan ploting menggunakan matplotlib
    colors=['red','blue','green', 'yellow', 'pink', 'grey', 'black', 'magenta', 'cyan', 'purple', 'burlywood', 'chartreuse', 'chocolate', 'indigo', 'orange']
    print(len(simpanan[0]))
    for i in range(15):
        for j in range(len(simpanan[i])):
            plot.scatter(simpanan[i][j][0],simpanan[i][j][1],color=colors[i])
    # print(simpanan[i][0])
    plot.show()
