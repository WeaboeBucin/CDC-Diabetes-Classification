import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Fungsi untuk melatih model LightGBM
# Memuat model dan skalar yang telah dilatih
with open('StandartScaler.pkl', 'rb') as file:
    StandardScaler = pickle.load(file)

with open('LightGBM.pkl', 'rb') as file:
    model_LightGBM = pickle.load(file)

# Fungsi utama Streamlit
def main():
    st.title("Klasifikasi dengan LightGBM")

    # Sidebar untuk pengguna memasukkan data
    st.sidebar.header("Masukkan Data")
    input_data = []
    st.sidebar.markdown("Tekanan Darah")
    input_dataTekananDarah = st.sidebar.number_input(f"0 Jika Tekanan Darah Rendah dan 1 Jika Tekanan Darah Tinggi", value=0, key = "TekananDarah")

    st.sidebar.markdown("Kolesterol")
    input_dataKolesterol = st.sidebar.number_input(f"0 Jika Kolesterol Rendah dan 1 Jika Kolesterol Tinggi", value=0, key = "Kolesterol")

    st.sidebar.markdown("Cek Kolesterol")
    input_dataCekKolesterol = st.sidebar.number_input(f"0 Jika Tidak Pernah Melakukan Cek Kolesterol Dalam 5 Tahun dan 1 Jika Pernah Melakukan Cek Kolesterol Dalam 5 Tahun", value=0, key = "CekKolesterol")

    st.sidebar.markdown("Berat Badan")
    input_dataBeratBadan = st.sidebar.number_input(f"Masukkan Berat Badan Dalam Satuan Kilo Gram", value=0, key = "BeratBadan")

    st.sidebar.markdown("Stroke")
    input_dataStroke = st.sidebar.number_input(f"0 Jika Tidak Memiliki Stroke dan 1 Jika Memiliki Stroke", value=0, key = "Stroke")

    st.sidebar.markdown("Penyakit Jantung")
    input_dataPenyakitJantung = st.sidebar.number_input(f"0 Jika Tidak Memiliki Penyakit Jantung dan 1 Jika Memiliki Penyakit Jantung", value=0, key = "PenyakitJantung")

    st.sidebar.markdown("Aktivitas Fisik")
    input_dataAktivitasFisik = st.sidebar.number_input(f"0 Jika Tidak Memiliki Aktivitas fisik selama 30 hari terakhir dan 1 Jika Memiliki", value=0, key = "AktivitasFisik")

    st.sidebar.markdown("Konsumsi Buah")
    input_dataKonsumsiBuah = st.sidebar.number_input(f"0 Jika Tidak Mengkonsumsi Buah Lebih Dari 1 Per Hari dan 1 Jika Iya", value=0, key = "KonsumsiBuah")

    st.sidebar.markdown("Konsumsi Alkohol Berlebih")
    st.sidebar.markdown("Pria Dewasa Mengkonsumsi Lebih Dari 14 Minuman Per Minggu Dan Wanita Dewasa 7 Minuman Per Minggu ")
    input_dataKonsumsiAlkoholBerlebih = st.sidebar.number_input(f"0 Jika Tidak dan 1 Jika Iya", value=0, key = "KonsumsiAlkoholBerlebih")

    st.sidebar.markdown("Kesulitan Bertemu Dokter")
    st.sidebar.markdown("Kesulitan Bertemu Dokter Dalam 12 Bulan Terakhir Karena Kendala Biaya")
    input_dataKesulitanBertemuDokter = st.sidebar.number_input(f"0 Jika Tidak dan 1 Jika Iya", value=0, key = "KesulitanBertemuDokter")

    st.sidebar.markdown("Kesehatan Tubuh")
    st.sidebar.markdown("Deskripsikan Kesehatan Tubuhmu Dengan Skala 1-5")
    input_dataKesehatanTubuh = st.sidebar.number_input(f"1 Jika Luarbiasa Baik, 2 Jika Sangat Baik, 3 Jika Baik, 4 Jika Biasa dan 5 Jika Buruk", value=0, key = "KesehatanTubuh")

    st.sidebar.markdown("Kesehatan Mental")
    st.sidebar.markdown("Deskripsikan Kesehatan Mentalmu, Termasuk stres, depresi, dan kesulitan dalam kontrol emosi untuk berapa hari dalam kurun 30 hari terakhir ini mentalmu kurang baik")
    input_dataKesehatanMental = st.sidebar.number_input(f"Skala dari 1 - 30 Hari", value=0, key = "KesehatanMental")

    st.sidebar.markdown("Kesehatan Fisik")
    st.sidebar.markdown("Deskripsikan Kesehatan Fisikmu, Termasuk penyakit fisik dan cidera untuk berapa hari dalam kurun 30 hari terakhir ini fisikmu kurang baik")
    input_dataKesehatanFisik = st.sidebar.number_input(f"Skala dari 1 - 30 Hari", value=0, key = "KesehatanFisik")

    st.sidebar.markdown("Kesulitan Berjalan")
    st.sidebar.markdown("Apakah Kamu Memiliki Kesulitan Dalam Berjalan Secara Serius")
    input_dataKesulitanBerjalan = st.sidebar.number_input(f"0 Jika Tidak dan 1 Jika Iya", value=0, key = "KesulitanBerjalan")

    st.sidebar.markdown("Jenis Kelamin")
    input_dataJenisKelamin = st.sidebar.number_input(f"0 Jika Perempuan dan 1 Jika Laki-laki", value=0, key = "JenisKelamin")

    st.sidebar.markdown("Umur")
    st.sidebar.markdown("Ada 13 Kategori Umur Yaitu : 1 Untuk Umur 18 - 24, 2 Untuk Umur 25 - 29, 3 Untuk Umur 30 - 34, 4 Untuk Umur 35 - 39, 5 Untuk Umur 40 - 44, 6 Untuk Umur 45 - 49, 7 Untuk Umur 50 - 54, 8 Untuk Umur 55 - 59, 9 Untuk Umur 60 - 64, 10 Untuk Umur 65 - 69, 11 Untuk Umur 70 - 74, 12 Untuk Umur 75 - 79, 13 Untuk Umur 80 Atau Lebih")
    input_dataUmur = st.sidebar.number_input(f"Skala Dari 1 - 13", value=0, key = "Umur")

    st.sidebar.markdown("Edukasi")
    st.sidebar.markdown("Ada 6 Kategori Yaitu : 1 Untuk Tidak Pernah Sekolah Atau Hanya Sampai TK Saja, 2 Untuk Siswa SD-SMP, 3 Untuk Siswa SMA, 4 Untuk Lulus SMA, 5 Untuk Mahasiswa Kuliah, 6 Untuk Lulus Dari Perkuliahan Atau Lebih")
    input_dataEdukasi = st.sidebar.number_input(f"Skala Dari 1 - 6", value=0, key = "Edukasi")

    st.sidebar.markdown("Penghasilan")
    st.sidebar.markdown("""Ada 8 Kategori Umur Yaitu : 1 Untuk Penghasilan Dibawah \\$10.000, 2 Untuk Penghasilan \\$10.000 - \\$15.000, 3 Untuk Penghasilan \\$15.000 - \\$20.000, 4 Untuk Penghasilan \\$20.000 - \\$25.000, 5 Untuk Penghasilan \\$25.000 - \\$35.000, 6 Untuk Penghasilan \\$35.000 - \\$50.000, 7 Untuk Penghasilan \\$50.000 - \\$75.000, 8 Untuk Penghasilan \\$75.000 atau lebih""")
    input_dataPenghasilan = st.sidebar.number_input(f"Skala Dari 1 - 8", value=0, key = "Penghasilan")

    # Menampilkan data yang dimasukkan pengguna
    input_dataset = [[input_dataTekananDarah,input_dataKolesterol,input_dataCekKolesterol,input_dataBeratBadan,input_dataStroke,input_dataPenyakitJantung,input_dataAktivitasFisik,input_dataKonsumsiBuah,input_dataKonsumsiAlkoholBerlebih,input_dataKesulitanBertemuDokter,input_dataKesehatanTubuh,input_dataKesehatanMental,input_dataKesehatanFisik,input_dataKesulitanBerjalan,input_dataJenisKelamin,input_dataUmur,input_dataEdukasi,input_dataPenghasilan]]

    input_df = pd.DataFrame({"HighBP": [input_dataTekananDarah],"HighChol": [input_dataKolesterol],"CholCheck": [input_dataCekKolesterol],"BMI": [input_dataBeratBadan],"Stroke": [input_dataStroke],"HeartDiseaseorAttack": [input_dataPenyakitJantung],"PhysActivity": [input_dataAktivitasFisik],"Fruits": [input_dataKonsumsiBuah],"HvyAlcoholConsump": [input_dataKonsumsiAlkoholBerlebih],"NoDocbcCost": [input_dataKesulitanBertemuDokter],"GenHlth": [input_dataKesehatanTubuh],"MentHlth": [input_dataKesehatanMental],"PhysHlth": [input_dataKesehatanFisik],"DiffWalk": [input_dataKesulitanBerjalan],"Sex": [input_dataJenisKelamin],"Age": [input_dataUmur],"Education": [input_dataEdukasi],"Income": [input_dataPenghasilan]})
    st.subheader("Data yang Dimasukkan")
    st.write(input_df)

    # Melatih model jika tombol ditekan
    if st.sidebar.button("Train Model"):

        # Latih model
        data_normal = StandardScaler.transform(input_df)

        # Lakukan prediksi
        prediction = model_LightGBM.predict(data_normal)

        st.subheader("Hasil Prediksi")
        st.write("Prediksi Kelas:", prediction)

if __name__ == "__main__":
    main()
