# Laporan Proyek Machine Learning - Ahmad Chandra Maulana

## Project Overview

Proyek ini membangun sistem rekomendasi anime yang memberikan hasil berupa daftar anime yang memiliki kemiripan (output) dengan genre anime yang diberikan (input). Sistem rekomendasi anime ini dibangun beradasarkan kemiripan genre suatu anime dan rating dari para pengguna. Genre membantu mengidentifikasi jenis film yang diminati pengguna, seperti action, slice of life, drama, dll, sehingga sistem dapat menawarkan pilihan yang sesuai dengan selera mereka. Sementara itu, content ratings dari user memastikan bahwa rekomendasi anime sesuai dengan preferensi  sensitivitas terhadap konten tertentu, menjaga pengalaman menonton yang nyaman dan aman.  Dengan mempertimbangkan kedua aspek ini, sistem rekomendasi menjadi lebih akurat, personal, dan bertanggung jawab, meningkatkan kepuasan pengguna secara keseluruhan.

### Background
Terdapat ribuan anime yang beredar di berbagai platform. Dengan banyaknya anime ini membuat para penonton kesulitan memilih anime yang sesuai dengan kesuakaannya. Sehingga, dengan adanya sistem rekomendasi, hal ini akakn membantu para penonton untuk memilih anime untuk ditonton berdarkan kemiripan genre dan rating para penonton lainnya.

## Business Understanding

Dalam merumuskan proyek ini dibangun penyataan masalah, tujuan, dan pernyataan solusi seperti di bawah ini.

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah rincian masalah:
* Berdasarkan data anime-anime yang ada, bagaimana membuat sistem rekomendasi anime yang dipersonalisasi dengan teknik content-based filtering?
* Dengan data user rating  pada anime, bagaimana perusahaan dapat merekomendasikan anime lain yang mungkin disukai dan belum pernah ditonton oleh penonton dengan collaborative filtering?

### Goals

Untuk menjawab pertanyaan masalah di atas, maka tujuan dari proyek ini adalah:
* Menghasilkan sejumlah rekomendasi anime yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
* Mampu mengembangkan model yang menghasilkan sejumlah rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik collaborative filtering dengan model yang mencapai metrik root mean square dibawah 0.15


### Solution statements

Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya sebagai berikut:
- Mengolah data teks berkaitan dengan informasi anime menjadi numerikal yang bisa dihitung nilai kemiripannya
- membuat sistem rekomendasi collaborative filtering berdasarkan data rating penonton.

## Data Understanding
Dataset yang digunakan pada proyek kali ini dibuat oleh CooperUnion yang terakhir diperbarui ke Kaggle pada tahun 2016. Sumber dataset: [Anime Recommendations Database
](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data). Sumber data ini terdiri atas 2 set, anime dan rating. 

Pada dataset anime terdiri atas 12294 baris dan 7 kolom data. Kondisi khusus dari data:
- memiliki jenis tipe data yang beragam untuk kolom-kolom yang ada diantaranya float64, int64, dan object
- Dataset anime terdiri atas 7 variabel. 3 variabel merupakan variabel numerik (anime_id dan member memiliki tipe data int64 sedangkan rating memiliki tipedata float), sedangkan sisanya merupakan variabel kategorikal.
- terdapat data yang tidak memiliki kosong
- tidak terdapat data duplikat

Pada dataset rating terdiri atas 7813737 baris dan 3 kolom data. tidak ada kondisi khusus dari data karena tidak memiliki nilai kosong.


Pada proyek ini, fitur yang digunakan adalah sebagai berikut beserta alasan:

Data anime
|Nama|Jenis|Tipe Data|Alasan|Jumlah Nilai Unik|
|----|-----|---------|------|-----------------|
|anime_id|independent|int64|id judul anime. diperlukan untuk mencari film-film yang mirip dan mengembalikan nilai dalam bentuk id anime|12294|
|name|dependent|object|judul anime|12292|
|genre|dependent|object|genre anime|3264|
|type|dependent|object|tipe anime (TV, Movie, OVA, ONA, dll)|6|
|episodes|dependent|object|Jumlah episode di anime tersebut|187|
|rating|dependent|float64|rata-rata rating anime dengan skala 10|598|
|members|dependent|int64|jumlah anggota dalam komunitas anime tersebut|6706|


Data rating
|Nama|Jenis|Tipe Data|Alasan|Jumlah Nilai Unik|
|----|-----|---------|------|-----------------|
|user_id|dependent|int64|id user|73515|
|anime_id|dependent|int64|id anime|11200|
|rating|independent|int64|rating yang diberikan pada anime|11|

Pada proyek ini,  tahapan EDA Univariate tidak dilakukan karena fitur utama untuk membuat model berbasis konten (content-based filtering) berdasarkan variabel genre yang memiliki tipe data list dalam bentuk string. Selain itu, satu anime dapat memiliki genre lebih dari satu. Hal ini akan menciptakan data overlapping memberikan interpretasi yang parsial pada interpretasi EDA.


## Data Preparation
### Content Based Filtering
### Handling missing value
Jumlah data bernilai kosong dicek menggunakan fungsi built-in df.isnull().sum() pada variabel anime. Terdapat 62 data kosong dalam variabel genre, 25 dalam variabel type dan 230  di rating. Data-data yang tidak memiliki nilai tersebut dihapus dari dataset.

### Handling duplicates
Jumlah data awal dan akhir setelah pengecekan duplikat tidak berubah. Tidak ada data duplikat dalam dataset.

### Content-based filtering preparation
Anime direkomendasikan berdasarkan kemiripan genre dan type (TV, Movie, OVA, dll) rata-rata yang diberikan oleh para penonton. Untuk menghitung cosine similarity, data perlu disiapkan dalam array vektor.

### TF-IDF Vectorizer(Manual) 
TF-IDF pada genre dilakukan dengan cara mengekstrak data teks dalam data genre dan dimasukkan ke dalam list. Jika genre belum ada di list, item genre akan ditambahkan ke dalam list genres. Setelah semua genre unik dipopulasikan ke dalam list, dataframe dibuat berdasarkan list genre sebagai kolom dengan masing-masing nilai data bernilai 0 (one-hot encoding). 

Dataframe digabungkan (concatenate) dengan anime_id agar jumlah baris data sebanyak baris data anime. Karena dataframe ini digabungkan, nilai data dalam kolom-kolom genre tidak memiliki nilai (NaN). Nilai NaN ini digantikan dengan nilai 0 agar ketika iterasi data genre, genre yang ada dalam teks dapat diberikan nilai 1 sehingga TF-IDF Vectorizer dapat tercapai. Setelah itu, data digabungkan dengan data anime sebelumnya berdasarkan anime_id dan diiterasikan data-data dalam kolom untuk melakukan TF-IDF pada kolom-kolom yang telah dilakukan one-hot encoding. TF-IDF Vectorizer selesai

untuk vektorisasi data type, data rating dilakukan one-hot encoding. Setelah itu, data type dan rating dihapus. Data sudah siap digunakan untuk analisis cosine similariti.


Secara ringkas, Berikut merupakan aktifitas yang dilakukan pada tahap Data Preparation:

Pada Data anime
|Aktifitas|Alasan|Ukuran dataset semula|Ukuran dataset sesudah preproses|
|---------|------|---------------------|--------------------------------|
|Mengekstrak data-data genre yang merupakan list menjadi array unik|Nilai pada data genre merupakan list, sehingga perlu dibuat sebagai one hot encoding dari masing-masing genre|(12017,7)|(12017,7)|
|membuat dataFrame hasil one-hot encoding dari data hasil array unik list genre|Mempersiapkan one-hot encoding data agar data terbaca oleh model|(12017,7)|(12017,7)|
|menggabungkan anime id dengan hasil one-hot encoding|mempersiapkan data untuk vektorisasi|(0,0)|(12017,44)|
|mengisi data gabungan yang tidak memiliki nilai 0|mempermudah proses vektorisasi agar ketika iterasi nilai-nilai pada genre, genre hasil one-hot encoding dapat diberikan nilai 1|(12017,44)|(12017,44)|
|menggabungkan dataframe one-hot encoding dengan data anime awal dengan basis anime_id sebagai jembatan|persiapan iterasi genre dengan one-hot encoding|(12017,44)|(12017,50)|
|melakukan iterasi pemecahan data-data pada genre dan dicocokkan dan menempatkan nilai 1 pada genre hasil one-hot encoding|melengkapi nilai data pada one-hot encoding genre|(12017,50)|(12017,50)|
|Melakukan one-hot encoding pada variabel type|Fitur type ditambahkan agar model dapat memberikan kemiripan yang lebih baik|(12017,50)|(12017,56)|
|Menghapus kolom type, member, ratings|data member dan ratings tidak diperlukan dan kolom type sudah direpresentasikan dengan one-hot encoding|(12017,56)|(12017,53)|

##  Collaborative Filtering

### Handling missing value
Dalam data user, terdapat penonton yang tidak memberikan penilaian pada anime yang ditonton dengan nilai -1. Data-data ini dihilangkan. Selain itu, karena terdapat anime-anime yang terhapus di dataset anime yang telah dipersiapkan. data anime_id yang tidak ada dalam dataset anime pada dataset ini juga dihapus.

### Handling duplicates
data duplikat dicek dengan user.drop_duplicates() dan terdapat 1 data duplikat dalam dataset.

### Encoding anime_id dan user_id
Encoding dilakukan dengan membuat urutan dictionary anime_id dan user_id sekaligus anti-decoding nya. Hal ini untuk membuat model mempelajari data lebih efektif

### train,test data split
Data user yang telah dipersiapkan dipisahkan menjadi data training dan data test untuk pemodelan sistem rekomendasi. data user_id dan anime_id digunakan sebagai fitur, sedangkan rating digunakan sebagai target dalam pemodelan ini. Setelah data dipisah menjadi train dan test menggunakan library sklearn, terdapat 5069716 data training dan 1267429 data testing.

Secara ringkas:
Pada data rating
|Aktifitas|Alasan|Ukuran dataset semula|Ukuran dataset sesudah preproses|
|---------|------|---------------------|--------------------------------|
|Mengeliminisasi user yang tidak memberikan penilaian|user yang tidak memberikan penilaian dapat merusak model|(7813737,3)|(6337241,3)|
|Mengeliminisasi anime_id yang tidak ada di dataset anime penilaian|menghindari error ketika pemilihan anime yang belum ditonton pada proses prediksi.|(6337241,3)|(6337146,3)|
|Mengeliminisasi data duplikat|agar pemodelan menjadi lebih handal|(6337146,3)|(6337145,3)|
|Melakukan encoding pada data anime_id dan user_id|untuk membuat model mempelajari data lebih efektif|(6337145,3)|(6337145,3)|



## Modeling -- content-based filtering

Pemodelan yang dipilih pada proyek ini adalah content-based filtering menggunakan cosine similarity karena implementasi content-based filtering dengan cosine similarity dapat cepat menghitung kesamaan antar film-film yang direpresentasikan dalam ruang vektor, sehingga sistem rekomendasi memberikan hasil dengan waktu respons yang cepat, berbeda jika membuat model sistem rekomendasi menggunakan deep learning yang membutuhkan biaya tinggi.

Yang dilakukan pada tahap ini diantaranya:
- Menghitung nilai kemiripan antar baris data film menggunakan cosine similariy lalu menyimpannya dalam bentuk dataframe. jika nilai kemiripan mendekati 1 berarti dua item memiliki banyak kemiripan. nilai kemiripan mendekati 0 berarti dua item tidak memiliki banyak kemiripan. dan nilai kemiripan mendekati -1 berarti 2 item saling berlawanan.
- Membuat fungsi get_recommendation() untuk mengeluarkan daftar nama-nama film yang disertai dengan urutan film yang paling mirip yang memiliki kemiripan dengan nama film yang dimasukkan sebagai parameter. Cara kerja dari fungsi ini adalah yang pertama memanfaatkan fungsi argpartition (mengembalikan indeks yang akan mempartisi array dalam cara tertentu sehingga elemen pada indeks yang dihasilkan memenuhi kriteria tertentu) untuk mengambil sejumlah nilai k tertinggi dari similarity data. Kemudian mengambil data dari tingkat kesamaan tertinggi ke terendah lalu dimasukkan ke dalam variabel closest yang menampung baris data film yang memiliki kemiripan tinggi dan yang terakhir ialah menghapus movie_title yang dicari menggunakan fungsi drop() agar tidak muncul dalam daftar rekomendasi karena sesama nilai akan menghasilkan nilai kesamaan tertinggi yaitu 1. Berikut merupakan penjelasan parameter dari fungsi get_recommendations() adalah sebagai berikut:

|nama|deskripsi|tipe data|
|----|---------|---------|
|anime_title|judul anime yang akan dicari judul-judul lain yang mirip|str|
|similarity_data|dataframe yang berisi nilai cosine similarity setiap anime yang ada dalam dataset|object|
|items|dataframe yang asli untuk mendapatkan nilai sebenarnya|object|
|k|banyaknya jumlah film yang mirip|int|

Berikut hasilnya yang diperoleh:

**Nama anime yang dimasukkan**
<img width="1223" alt="Screenshot 2024-12-08 at 07 28 48" src="https://github.com/user-attachments/assets/377a19be-608a-4ae6-83ba-8540ae237ab9">


**Daftar film yang direkomendasikan**
<img width="1223" alt="Screenshot 2024-12-08 at 08 00 51" src="https://github.com/user-attachments/assets/e96e536c-8149-40b6-87fc-6252a6f5b8c0">

Berdasarkan pengamatan input dan output diatas menunjukkan bahwa daftar film yang direkomendasikan memiliki genre yang hampir semuanya sama.

## Modeling -- Collaborative filtering
Collaborative filtering ini menggunakan RecommenderNet. Model ini menghitung skor kecocokan antara penonton dan anime dengan teknik embedding. Pertama, proses embedding dilakukan terhadap data user dan anime. Selanjutnya,  operasi perkalian dot product dilakukan diantara embedding user dan anime. Selain itu, bias ditambahkan kepada setiap user dan anime. Skor kecocokan berada dalam skala [0,1] dengan fungsi aktivasi sigmoid. 

Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.

Dalam pemodelan, digunakan data train, test split sebanyak 80% dan 20%.

Untuk mendapatkan rekomendasi buku, untuk sementara sampel user diambil secara acak dan definisikan variabel anime_not_watched yang merupakan anime-anime yang belum pernah ditonton oleh penonton , data-data dalam anime_not_watched inilah yang akan menjadi anime_not_watched yang rekomendasikan. Variabel anime_not_watched diperoleh dengan menggunakan operator bitwise (~) pada variabel anime_watched_by_user. Sebelumnya, penonton telah memberikan rating pada anime-anime yang mereka tonton. Rating ini akan digunakan untuk membuat rekomendasi anime yang mungkin cocok untuk penonton. Kemudian, untuk memperoleh rekomendasi anime, menggunakan fungsi model.predict() dari library Keras. Hasil rekomendasinya adalah seperti berikut.

<img width="1014" alt="Screenshot 2024-12-11 at 06 40 43" src="https://github.com/user-attachments/assets/08172215-5dd7-4b08-9a61-c9a9f18b0c9f">



## Evaluation
### Content-Based filtering
Padal Content-based filtering, Precision digunakan sebagai metrics dalam pemodelan. Precision dapat dihitung dengan membandingkan jumlah anime yang relevan dengan jumlah anime yang direkomendasikan.

Dalam kasus ini, anime Toaru Kagaku no Railgun digunakan sebagai sampel. Anime ini merupakan TV series (type) dengan genre Action, Sci-fi, Super Power

<img width="551" alt="Screenshot 2024-12-11 at 10 09 13" src="https://github.com/user-attachments/assets/4adb4554-8be0-4d5b-b779-29f3317bf652">

Berikut ini adalah rekomendasi yang diberikan oleh model sistem rekomendasi
<img width="565" alt="Screenshot 2024-12-11 at 10 11 47" src="https://github.com/user-attachments/assets/6d5ba2b0-ba5c-4924-9602-1e9978959e71">

Berdasarkan hasil sistem rekomendasi, seluruh anime merupakan TV series (type) dan genre anime semuanya memiliki genre Action, Sci-fi, dan Super Power. Selain itu, pada urutan-urutan awal, genre anime yang direkomendasikan sama percis dengan genre anime yang telah ditonton, dilanjutkan dengan genre-genre tambahan tanpa mengurangi relevansi dari anime yang direkomendasikan. Sehingga berdasarkan hasil rekomendasi yang diberikan, 10 dari 10 data memberikan rekomendasi anime yang relevan, metrik precision dalam model content-based filtering ini adalah 100%


### Collaborative Filtering
Model Collaborative filtering dibangung dengan RecommenderNet yang memanfaatkan embedding layer. Metrik yang digunakan adalah Root Mean Squared Error (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut,

<img width="237" alt="Screenshot 2024-12-11 at 06 43 16" src="https://github.com/user-attachments/assets/930e7853-dc25-4cdc-8937-c000de5c6a16">

Di mana, nilai  n  merupakan jumlah dataset, nilai  y_i  adalah nilai sebenarnya, dan  y_pred  yaitu nilai prediksinya terdahap  i  sebagai urutan data dalam dataset.

Semakin kecil nilai RMSE, semakin baik model memprediksi. 

<img width="676" alt="Screenshot 2024-12-11 at 06 45 38" src="https://github.com/user-attachments/assets/046bcc7f-a53f-4bd7-90bc-d391a0f9c17d">

Berdasarkan plot model RMSE, model tidak menunjukkan overfitting atau underfitting. Overfitting biasanya akan terjadi jika nilai RMSE pada data pelatihan terus menurun sementara RMSE pada data validasi meningkat. Sebaliknya, Underfitting biasanya terjadi jika nilai RMSE baik pada data pelatihan maupun data validasi tinggi. Dalam kasus ini, baik RMSE pada data pelatihan maupun data validasi menurun seiring dengan berjalannya epoch dan RMSE pada data validasi juga mencapai tingkat yang rendah, yang menunjukkan bahwa model secara umum berhasil melakukan prediksi dengan baik. Root mean square error model sekitar 0.13. Nilai ini sangat bagus untuk pemodelan sistem rekomendasi


## Penutup
- model content-based filtering memberikan daftar anime rekomendsasi yang mirip berdasarkan genre dan tipe anime yang ditonton oleh penonton. Hal ini dapat memberikan pilihan-pilihan anime kepada penonton lebih terpersonalisasi. 
- model collaborative filtering berhasil memberikan rekomendasi berdasarkan user dan anime yang ditonton oleh penonton lain dengan baik (model memiliki root mean square error di bawah 0.15)




