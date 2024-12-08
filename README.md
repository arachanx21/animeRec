# Laporan Proyek Machine Learning - Ahmad Chandra Maulana

## Project Overview

Proyek ini membangun sistem rekomendasi anime yang memberikan hasil berupa daftar anime yang memiliki kemiripan (output) dengan genre anime yang diberikan (input). Sistem rekomendasi anime ini dibangun beradasarkan kemiripan genre suatu anime dan rating dari para pengguna. Genre membantu mengidentifikasi jenis film yang diminati pengguna, seperti action, slice of life, drama, dll, sehingga sistem dapat menawarkan pilihan yang sesuai dengan selera mereka. Sementara itu, content ratings dari user memastikan bahwa rekomendasi anime sesuai dengan preferensi  sensitivitas terhadap konten tertentu, menjaga pengalaman menonton yang nyaman dan aman.  Dengan mempertimbangkan kedua aspek ini, sistem rekomendasi menjadi lebih akurat, personal, dan bertanggung jawab, meningkatkan kepuasan pengguna secara keseluruhan.

### Background
Terdapat ribuan anime yang beredar di berbagai platform. Dengan banyaknya anime ini membuat para penonton kesulitan memilih anime yang sesuai dengan kesuakaannya. Sehingga, dengan adanya sistem rekomendasi, hal ini akakn membantu para penonton untuk memilih anime untuk ditonton berdarkan kemiripan genre dan rating para penonton lainnya.

## Business Understanding

Dalam merumuskan proyek ini dibangun penyataan masalah, tujuan, dan pernyataan solusi seperti di bawah ini.

### Problem Statements

Berdasarkan latar belakang di atas, rincian masalahnya adalah **bagaimana memanfaatkan data amime agar dapat meningkatkan kepuasan penonton mengembangkan sistem rekomendasi film yang lebih relevan dengan genre kesukaan pengguna?**

### Goals

Untuk menjawab pertanyaan masalah di atas, maka tujuan dari proyek ini adalah **mampu mengembangkan model yang sepresisi mungkin dalam memberikan daftar film sesuai genre yang disukai oleh penonton dengan model yang berbasis dengan konten dan user (collaborative filtering) dengan model yang mencapai metrik root mean square dibawah 0.15%**

### Solution statements

Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya sebagai berikut:
- Mengolah data teks berkaitan dengan informasi film menjadi numerikal yang bisa dihitung nilai kemiripannya
- Menampilkan detail data yang diuji dengan hasilnya untuk validasi kebenaran dari nilai metriknya

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
<h3>Data anime</h3>
|Nama|Jenis|Tipe Data|Alasan|Jumlah Nilai Unik|
|----|-----|---------|------|-----------------|
|anime_id|independent|int64|id judul anime. diperlukan untuk mencari film-film yang mirip dan mengembalikan nilai dalam bentuk id anime|12294|
|name|dependent|object|judul anime|12292|
|genre|dependent|object|genre anime|3264|
|type|dependent|object|tipe anime (TV, Movie, OVA, ONA, dll)|6|
|episodes|dependent|object|Jumlah episode di anime tersebut|187|
|rating|dependent|float64|rata-rata rating anime dengan skala 10|598|
|members|dependent|int64|jumlah anggota dalam komunitas anime tersebut|6706|

<h3>Data rating</h3>
|Nama|Jenis|Tipe Data|Alasan|Jumlah Nilai Unik|
|----|-----|---------|------|-----------------|
|user_id|dependent|int64|id user|73515|
|anime_id|dependent|int64|id anime|11200|
|rating|independent|int64|rating yang diberikan pada anime|11|

Pada proyek ini,  tahapan EDA Univariate tidak dilakukan karena fitur utama untuk membuat model berbasis konten (content-based filtering) berdasarkan variabel genre yang memiliki tipe data list dalam bentuk string. Selain itu, satu anime dapat memiliki genre lebih dari satu. Hal ini akan menciptakan data overlapping memberikan interpretasi yang parsial pada interpretasi EDA.


## Data Preparation
Pada tahap ini perlu mempersiapkan data agar mudah diproses oleh model, apalagi jika masih mengandung kolom-kolom yang bertipe data object yang perlu dilakukan encoding untuk mengubahnya menjadi numerik supaya dapat dihitung nilai cosine similarity nya.

Berikut merupakan aktifitas lainnya dilakukan pada tahap ini:
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

Pada data rating
|Aktifitas|Alasan|Ukuran dataset semula|Ukuran dataset sesudah preproses|
|---------|------|---------------------|--------------------------------|
|Mengeliminisasi user yang tidak memberikan penilaian|user yang tidak memberikan penilaian dapat merusak model|(7813737,3)|(6337241,3)|
|Mengeliminisasi anime_id yang tidak ada di dataset anime penilaian|menghindari error ketika pemilihan anime yang belum ditonton pada proses prediksi.|(6337241,3)|(6337146,3)|
|Melakukan encoding pada data anime_id dan user_id|untuk membuat model mempelajari data lebih efektif|(6337146,3)|(6337146,3)|


**catatan**
- ***proyek ini tidak menggunakan vectorizer*** karena sudah terwakilili dengan hasil onehot encoding untuk kolom genre. hasil dari onehot encoding sudah seperti vektor yang dihasilkan dari binary count vectorizer. selain itu,  pada proyek kali ini, secara manual melakukan term-frequency karena dalam satu baris data memiliki nilai-nilai yang berbeda yang perlu diubah dulu agar dapat menjadi vektor.

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
Collaborative filtering menghitung skor kecocokan antara penonton dan anime dengan teknik embedding. Pertama, proses embedding dilakukan terhadap data user dan anime. Selanjutnya,  operasi perkalian dot product dilakukan diantara embedding user dan anime. Selain itu, bias ditambahkan kepada setiap user dan anime. Skor kecocokan berada dalam skala [0,1] dengan fungsi aktivasi sigmoid. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.

Dalam pemodelan, digunakan data train, test split sebanyak 80% dan 20%


## Evaluation


Catatan:
- solusi sudah menjawab problem statement karena telah membuat model untuk memberikan daftar film yang mirip dengan film yang dimasukkan
- sudah mencapai goals yang diharapkan karena berhasil membangun model yang memiliki presisi lebih dari 85%
- solusi yang direncakan berdampak pada hasil karena dapat model bisa berjalan dengan cepat dengan penentuan kolom genres dan content_ratings yang telah berbentuk binary count vector

## References


