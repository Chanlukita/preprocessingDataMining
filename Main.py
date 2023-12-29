import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from sklearn.metrics import (
    silhouette_score,
)


st.set_page_config(page_title="Segmentasi Customer")

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Segmentasi Customer</h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=["Home", "Test", "Information"],
    icons=["house", "file-earmark-arrow-up", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    st.markdown(
        """
    <div style="text-align: center;">
        <h1>SUKUN</h1>
    </div>
    """,
        unsafe_allow_html=True
    )
    # Membaca Dataset dari CSV
    df = pd.read_csv("Data Sukun Jual.csv", sep=';')
    # Mengganti nama kolom kdplg menjadi ID Pelanggan
    df.rename(columns={"kdplg": "ID Pelanggan"}, inplace=True)
    st.header("Dataset")
    st.dataframe(df)

    day = "2023-12-01"
    # Mengonversi string tanggal day menjadi objek pandas datetime.
    day = pd.to_datetime(day)
    # Mengatasi kesalahan konversi, dan nilai yang tidak dapat diubah menjadi tanggal akan diisi dengan NaT (Not a Time).
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    # Membuat DataFrame baru yang hanya berisi tahun 2020 - 2023.
    df_filtered = df[df['tanggal'].dt.year.between(2020, 2023)]

    # Recency (waktu terakhir pembelian).
    recency = df_filtered.groupby(["ID Pelanggan"]).agg(
        {"tanggal": lambda x: ((day - x.max()).days)})

    # Frequency (frekuensi pembelian).
    freq = df.drop_duplicates(subset="nota").groupby(
        ["ID Pelanggan"])[["nota"]].count()

    # Monetary (total nilai pembelian).
    df["total"] = df["jumlah"]*df["hgjual"]
    money = df.groupby(["ID Pelanggan"])[["total"]].sum()

    # Menggabungkan hasil Recency, Frequency, dan Monetary ke dalam DataFrame RFM.
    recency.columns = ["Recency"]
    freq.columns = ["Frequency"]
    money.columns = ["Monetary"]
    RFM = pd.concat([recency, freq, money], axis=1)
    st.header("RFM")
    st.write(RFM)

    # Normalisasi Data
    RFM = RFM.fillna(5)  # Mengisi nilai NaN dengan 5 pada DataFrame RFM.

    # menormalkan (scaling) data RFM sehingga memiliki skala yang seragam.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(RFM)

    # Menentukan Jumlah Kluster Optimal.
    inertia = []  # Elbow Method:
    for i in np.arange(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=20)
        kmeans.fit(scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(1, 11), inertia, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    st.header("Elbow Method")
    st.pyplot(fig)

    # Menambahkan elemen input untuk nilai kluster.
    num_clusters = st.number_input(
        "Masukan angka Kluster", min_value=2, max_value=9, value=4)

    # Proses clustering setelah menentukan jumlah kluster optimal menggunakan metode siku.
    kmeans = KMeans(n_clusters=num_clusters, random_state=20)
    kmeans.fit(scaled)
    RFM["Kluster"] = (kmeans.labels_ + 1)

    # Melakukan pengelompokkan (grouping) data berdasarkan kolom "Kluster"
    final = RFM.groupby(["Kluster"])[
        ["Recency", "Frequency", "Monetary"]].mean()

    st.header("Average RFM Values by Cluster")
    st.dataframe(final)

    # Visualisasi Hasil Clustering:
    st.header("Clustering Results")
    clustering_results = []
    for i in range(2, 10):
        if i == num_clusters:
            kmeans = KMeans(n_clusters=i, random_state=20)
            kmeans.fit(scaled)
            labels = kmeans.labels_

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
                4, 2, figsize=(30, 40))  # Membuat 6 petak agar diagram tidak bertabrakan

            # Scatter Plot
            scatter = ax1.scatter(
                scaled[:, 0], scaled[:, 1], c=labels, cmap='viridis')
            ax1.set_title(f"Scatter Plot for {i} Clusters")
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')

            # Menambahkan warna legenda pada scatter plot
            legend = ax1.legend(*scatter.legend_elements(),
                                title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Menambahkan penjelasan warna pada legenda
            legend.set_title("Clusters")
            for text, cluster in zip(legend.get_texts(), range(1, i + 1)):
                text.set_text(f"Cluster {cluster}")

            # Pengertian Warna
            ax2.axis('off')
            ax2.text(0.1, 0.5, 'Pengertian Warna:\nWarna merepresentasikan kluster pada Scatter Plot',
                     fontsize=12, ha='left', va='center')

            # Heatmap Korelasi
            correlation_matrix = RFM.corr()
            sns.heatmap(correlation_matrix, annot=True,
                        cmap='coolwarm', ax=ax3)
            ax3.set_title("Correlation Heatmap")

            # Diagram Batang
            cluster_counts = pd.Series(
                labels).value_counts().sort_index()
            ax4.bar(cluster_counts.index, cluster_counts.values,
                    color=plt.cm.viridis(np.linspace(0, 1, i)))
            ax4.set_title("Cluster Distribution")
            ax4.set_xlabel("Cluster Label")
            ax4.set_ylabel("Count")

            # Box Plot
            sns.boxplot(x='Kluster', y='Recency', data=RFM,
                        palette='viridis', ax=ax5)
            ax5.set_title("Box Plot of Recency by Cluster")

            # Pair Plot
            pair_plot = sns.pairplot(
                RFM, hue='Kluster', palette='viridis', diag_kind='kde', height=3)
            st.write(f"Kluster {i}")
            st.pyplot(pair_plot.fig)

            # Violin Plot
            sns.violinplot(x='Kluster', y='Monetary',
                           data=RFM, palette='viridis', ax=ax6)
            ax6.set_title("Violin Plot of Monetary by Cluster")

            # Standar Deviasi (Inertia) Plot
            inertias = []
            for n in range(1, i + 1):
                kmeans_n = KMeans(n_clusters=n, random_state=20)
                kmeans_n.fit(scaled)
                inertias.append(kmeans_n.inertia_)

            ax7.plot(range(1, i + 1), inertias, marker='o')
            ax7.set_title(f'Standard Deviation (Inertia) for {i} Clusters')
            ax7.set_xlabel('Number of Clusters')
            ax7.set_ylabel('Inertia')

            # Menampilkan Standar Deviasi di Subplot Ke-8
            stdev_value = kmeans.inertia_
            ax8.text(0.5, 0.5, f"Standard Deviation (Inertia): {stdev_value:.2f}",
                     fontsize=18, ha='center', va='center')
            ax8.axis('off')

            st.pyplot(fig)
            clustering_results.append(
                {"clusters": i, "labels": labels, "inertia": stdev_value})

    # Interpretasi Hasil Clustering
    def func(row):
        if row["Kluster"] == 1:
            return 'Kluster 1 (Bronze)'
        elif row["Kluster"] == 2:
            return 'Kluster 2 (Silver)'
        elif row["Kluster"] == 3:
            return 'Kluster 3 (Gold)'
        elif row["Kluster"] == 4:
            return 'Kluster 4 (Platinum)'
        elif row["Kluster"] == 5:
            return 'Kluster 5 (Diamond)'
        elif row["Kluster"] == 6:
            return 'Kluster 6 (Elite)'
        elif row["Kluster"] == 7:
            return 'Kluster 7 (Premier)'
        elif row["Kluster"] == 8:
            return 'Kluster 8 (Prestige)'
        elif row["Kluster"] == 9:
            return 'Kluster 9 (Royal)'
        else:
            return 'Tidak Diketahui'

    RFM['group'] = RFM.apply(func, axis=1)
    st.header("Hasil Kluster")
    st.write(RFM)

    # Visualisasi Distribusi Kluster
    colors = ["DarkRed", "DarkCyan", "DarkBlue", "Yellow"][:num_clusters]
    result = RFM.group.value_counts()

    # Menampilkan Plot Hasil Klaster
    fig_result, ax_result = plt.subplots(figsize=(10, 6))
    result.plot(kind="barh", color=colors)
    ax_result.set_title("Result")
    ax_result.set_xlabel("Count")
    ax_result.set_ylabel("Group")
    st.header("Result")
    st.pyplot(fig_result)

    silhouette_avg = silhouette_score(scaled, RFM["Kluster"])
    st.write("silhouette score: ", silhouette_avg)

    st.write(result)

    # Medeskripsikan kluster sesuai yang di input.
    st.subheader("Deskripsi Kluster")
    if num_clusters == 2:
        st.write(
            """
        - **Kluster 1:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, memiliki frekuensi pembelian yang sedang, dan memiliki nilai moneter yang tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan dan meningkatkan keterlibatan pelanggan, serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.

        - **Kluster 2:**
            - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.
        """
        )
    elif num_clusters == 3:
        st.write(
            """
        - **Kluster 1:**
            - Pelanggan dalam kluster ini adalah pelanggan yang sangat aktif dan sering berbelanja dengan nilai moneter yang signifikan. Fokus strategi pemasaran dapat ditempatkan untuk mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

        - **Kluster 2:**
            - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.

        - **Kluster 3:**
            -  Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lebih lama dan memiliki nilai moneter yang signifikan. Strategi pemasaran dapat difokuskan untuk mempertahankan dan meningkatkan keterlibatan pelanggan serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.
            """
        )
    elif num_clusters == 4:
        st.write(
            """
        - **Kluster 1:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lebih lama, memiliki frekuensi pembelian yang sedang, dan memiliki nilai moneter yang signifikan. Strategi pemasaran dapat difokuskan pada mempertahankan dan meningkatkan keterlibatan pelanggan, serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.

        - **Kluster 2:**
            - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.

        - **Kluster 3:**
            - Pelanggan dalam kluster ini adalah pelanggan yang baru-baru ini berinteraksi dengan bisnis, sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

        - **Kluster 4:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.
            """
        )
    elif num_clusters == 5:
        st.write(
            """
        - **Kluster 1:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lebih lama, memiliki frekuensi pembelian yang sedang, dan memiliki nilai moneter yang tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan dan meningkatkan keterlibatan pelanggan, serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.

        - **Kluster 2:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sangat sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

        - **Kluster 3:**
            - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.

        - **Kluster 4:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sering berbelanja, dan memiliki nilai moneter yang tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

        - **Kluster 5:**
            - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sangat sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.
            """
        )
    elif num_clusters == 6:
        st.write(
            """
        - **Kluster 1:**
            - Frekuensi Pembelian Rendah.
            - Nilai Moneter Tinggi.

        - **Kluster 2:**
            - Frekuensi Pembelian Sedang.
            - Nilai Moneter Tinggi.

        - **Kluster 3:**
            - Frekuensi Pembelian Tinggi.
            - Nilai Moneter Sangat Tinggi.

        - **Kluster 4:**
            - Frekuensi Pembelian Tinggi.
            - Nilai Moneter Tinggi.

        - **Kluster 5:**
            - Frekuensi Pembelian Rendah.
            - Nilai Moneter Rendah.

        - **Kluster 6:**
            - Frekuensi Pembelian Tinggi.
            - Nilai Moneter Sangat Tinggi.
        """
        )
    elif num_clusters == 7:
        st.write(
            """
        - **Kluster 1:**
            - Frekuensi Pembelian Rendah.
            - Nilai Moneter Tinggi.

        - **Kluster 2:**
            - Frekuensi Pembelian Tinggi.
            - Nilai Moneter Sangat Tinggi.

        - **Kluster 3:**
            - Frekuensi Pembelian Rendah.
            - Nilai Moneter Tinggi.

        - **Kluster 4:**
            - Frekuensi Pembelian Tinggi.
            - Nilai Moneter Tinggi.

        - **Kluster 5:**
            - Frekuensi Pembelian Sedang.
            - Nilai Moneter Tinggi.

        - **Kluster 6:**
            - Frekuensi Pembelian Tinggi.
            - Nilai Moneter Sangat Tinggi.

        - **Kluster 7:**
            - Frekuensi Pembelian Rendah.
            - Nilai Moneter Rendah.
        """
        )

    # Medeskripsikan kluster sesuai yang di input.
    st.subheader("Strategi Pemasaran")
    if num_clusters == 2:
        st.write(
            """
        1. **Kluster 1:**
        - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
        - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:** Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.

        2. **Kluster 2:**
        - Kampanye retargeting untuk meningkatkan keterlibatan.
        - Program insentif untuk mendorong peningkatan frekuensi pembelian.
        - **Kelemahan:** Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

        Penting untuk terus memantau dan mengevaluasi efektivitas strategi ini serta melakukan analisis lebih lanjut terkait dengan konteks bisnis dan tren pasar untuk memastikan relevansi strategi pemasaran seiring waktu.
        """
        )
    elif num_clusters == 3:
        st.write(
            """
        1. **Kluster 1:**
        - **Strategi Pemasaran:**
            - Program loyalitas eksklusif untuk meningkatkan retensi pelanggan.
            - Penawaran bundel atau diskon khusus untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

        2. **Kluster 2:**
        - **Strategi Pemasaran:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.
        - **Kelemahan:**
            - Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

        3. **Kluster 3:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.
        """
        )
    elif num_clusters == 4:
        st.write(
            """
        1. **Kluster 1:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.

        2. **Kluster 2:**
        - **Strategi Pemasaran:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.
        - **Kelemahan:**
            - Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

        3. **Kluster 3:**
        - **Strategi Pemasaran:**
            - Program loyalitas eksklusif untuk meningkatkan retensi pelanggan.
            - Penawaran bundel atau diskon khusus untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

        4. **Kluster 4:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

        Penting untuk terus memantau dan mengevaluasi efektivitas strategi ini serta melakukan analisis lebih lanjut terkait dengan konteks bisnis dan tren pasar untuk memastikan relevansi strategi pemasaran seiring waktu.
        """
        )
    elif num_clusters == 5:
        st.write(
            """
        1. **Kluster 1:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.

        2. **Kluster 2:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

        3. **Kluster 3:**
        - **Strategi Pemasaran:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.
        - **Kelemahan:**
            - Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

        4. **Kluster 4:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

        5. **Kluster 5:**
        - **Strategi Pemasaran:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        - **Kelemahan:**
            - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

        Penting untuk terus memantau dan mengevaluasi efektivitas strategi ini serta melakukan analisis lebih lanjut terkait dengan konteks bisnis dan tren pasar untuk memastikan relevansi strategi pemasaran seiring waktu.
        """
        )

    elif num_clusters == 6:
        st.write(
            """
        1. **Kluster 1:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.

        2. **Kluster 2:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        3. **Kluster 3:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        4. **Kluster 4:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        5. **Kluster 5:**
            - Meningkatkan keterlibatan dengan kampanye personalisasi.
            - Program insentif untuk meningkatkan frekuensi pembelian.

        6. **Kluster 6:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
        """
        )
    elif num_clusters == 7:
        st.write(
            """
        1. **Kluster 1:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.

        2. **Kluster 2:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        3. **Kluster 3:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.

        4. **Kluster 4:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        5. **Kluster 5:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        6. **Kluster 6:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

        7. **Kluster 7:**
            - Meningkatkan keterlibatan dengan kampanye personalisasi.
            - Program insentif untuk meningkatkan frekuensi pembelian.
        """
        )

elif selected == "Test":

    # Upload Data Test
    st.markdown(
        """
    <div style="text-align: center;">
        <h1>Upload CSV</h1>
    </div>
    """,
        unsafe_allow_html=True
    )
    # Upload file yg bertipe CSV
    data_file = st.file_uploader("", type=["CSV"])

    # Detail File
    if data_file is not None:
        file_details = {"filename": data_file.name,
                        "filetype": data_file.type,
                        "filesize": data_file.size}
        st.write(file_details)

       # Membaca Dataset dari CSV
        df = pd.read_csv(data_file, sep=';')
        # Mengganti nama kolom kdplg menjadi ID Pelanggan
        df.rename(columns={"kdplg": "ID Pelanggan"}, inplace=True)
        st.header("Dataset")
        st.dataframe(df)

        day = "2023-12-01"
        # Mengonversi string tanggal day menjadi objek pandas datetime.
        day = pd.to_datetime(day)
        # Mengatasi kesalahan konversi, dan nilai yang tidak dapat diubah menjadi tanggal akan diisi dengan NaT (Not a Time).
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        # Membuat DataFrame baru yang hanya berisi tahun 2020 - 2023.
        df_filtered = df[df['tanggal'].dt.year.between(2020, 2023)]

        # Recency (waktu terakhir pembelian).
        recency = df_filtered.groupby(["ID Pelanggan"]).agg(
            {"tanggal": lambda x: ((day - x.max()).days)})

        # Frequency (frekuensi pembelian).
        freq = df.drop_duplicates(subset="nota").groupby(
            ["ID Pelanggan"])[["nota"]].count()

        # Monetary (total nilai pembelian).
        df["total"] = df["jumlah"]*df["hgjual"]
        money = df.groupby(["ID Pelanggan"])[["total"]].sum()

        # Menggabungkan hasil Recency, Frequency, dan Monetary ke dalam DataFrame RFM.
        recency.columns = ["Recency"]
        freq.columns = ["Frequency"]
        money.columns = ["Monetary"]
        RFM = pd.concat([recency, freq, money], axis=1)
        st.header("RFM")
        st.write(RFM)

        # Normalisasi Data
        RFM = RFM.fillna(5)  # Mengisi nilai NaN dengan 5 pada DataFrame RFM.

        # menormalkan (scaling) data RFM sehingga memiliki skala yang seragam.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(RFM)

        # Menentukan Jumlah Kluster Optimal.
        inertia = []  # Elbow Method:
        for i in np.arange(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=20)
            kmeans.fit(scaled)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(1, 11), inertia, marker='o')
        ax.set_title('Elbow Method for Optimal k')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        ax.grid(True)
        st.header("Elbow Method")
        st.pyplot(fig)

        # Menambahkan elemen input untuk nilai kluster.
        num_clusters = st.number_input(
            "Masukan angka Kluster", min_value=2, max_value=9, value=4)

        # Proses clustering setelah menentukan jumlah kluster optimal menggunakan metode siku.
        kmeans = KMeans(n_clusters=num_clusters, random_state=20)
        kmeans.fit(scaled)
        RFM["Kluster"] = (kmeans.labels_ + 1)

        # Melakukan pengelompokkan (grouping) data berdasarkan kolom "Kluster"
        final = RFM.groupby(["Kluster"])[
            ["Recency", "Frequency", "Monetary"]].mean()

        st.header("Average RFM Values by Cluster")
        st.dataframe(final)

        # Visualisasi Hasil Clustering:
        st.header("Clustering Results")
        clustering_results = []
        for i in range(2, 10):
            if i == num_clusters:
                kmeans = KMeans(n_clusters=i, random_state=20)
                kmeans.fit(scaled)
                labels = kmeans.labels_

                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
                    4, 2, figsize=(30, 40))  # Membuat 6 petak agar diagram tidak bertabrakan

                # Scatter Plot
                scatter = ax1.scatter(
                    scaled[:, 0], scaled[:, 1], c=labels, cmap='viridis')
                ax1.set_title(f"Scatter Plot for {i} Clusters")
                ax1.set_xlabel('Feature 1')
                ax1.set_ylabel('Feature 2')

                # Menambahkan warna legenda pada scatter plot
                legend = ax1.legend(*scatter.legend_elements(),
                                    title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

                # Menambahkan penjelasan warna pada legenda
                legend.set_title("Clusters")
                for text, cluster in zip(legend.get_texts(), range(1, i + 1)):
                    text.set_text(f"Cluster {cluster}")

                # Pengertian Warna
                ax2.axis('off')
                ax2.text(0.1, 0.5, 'Pengertian Warna:\nWarna merepresentasikan kluster pada Scatter Plot',
                         fontsize=12, ha='left', va='center')

                # Heatmap Korelasi
                correlation_matrix = RFM.corr()
                sns.heatmap(correlation_matrix, annot=True,
                            cmap='coolwarm', ax=ax3)
                ax3.set_title("Correlation Heatmap")

                # Diagram Batang
                cluster_counts = pd.Series(
                    labels).value_counts().sort_index()
                ax4.bar(cluster_counts.index, cluster_counts.values,
                        color=plt.cm.viridis(np.linspace(0, 1, i)))
                ax4.set_title("Cluster Distribution")
                ax4.set_xlabel("Cluster Label")
                ax4.set_ylabel("Count")

                # Box Plot
                sns.boxplot(x='Kluster', y='Recency', data=RFM,
                            palette='viridis', ax=ax5)
                ax5.set_title("Box Plot of Recency by Cluster")

                # Pair Plot
                pair_plot = sns.pairplot(
                    RFM, hue='Kluster', palette='viridis', diag_kind='kde', height=3)
                st.write(f"Kluster {i}")
                st.pyplot(pair_plot.fig)

                # Violin Plot
                sns.violinplot(x='Kluster', y='Monetary',
                               data=RFM, palette='viridis', ax=ax6)
                ax6.set_title("Violin Plot of Monetary by Cluster")

                # Standar Deviasi (Inertia) Plot
                inertias = []
                for n in range(1, i + 1):
                    kmeans_n = KMeans(n_clusters=n, random_state=20)
                    kmeans_n.fit(scaled)
                    inertias.append(kmeans_n.inertia_)

                ax7.plot(range(1, i + 1), inertias, marker='o')
                ax7.set_title(f'Standard Deviation (Inertia) for {i} Clusters')
                ax7.set_xlabel('Number of Clusters')
                ax7.set_ylabel('Inertia')

                # Menampilkan Standar Deviasi di Subplot Ke-8
                stdev_value = kmeans.inertia_
                ax8.text(0.5, 0.5, f"Standard Deviation (Inertia): {stdev_value:.2f}",
                         fontsize=18, ha='center', va='center')
                ax8.axis('off')

                st.pyplot(fig)
                clustering_results.append(
                    {"clusters": i, "labels": labels, "inertia": stdev_value})

        # Interpretasi Hasil Clustering
        def func(row):
            if row["Kluster"] == 1:
                return 'Kluster 1 (Bronze)'
            elif row["Kluster"] == 2:
                return 'Kluster 2 (Silver)'
            elif row["Kluster"] == 3:
                return 'Kluster 3 (Gold)'
            elif row["Kluster"] == 4:
                return 'Kluster 4 (Platinum)'
            elif row["Kluster"] == 5:
                return 'Kluster 5 (Diamond)'
            elif row["Kluster"] == 6:
                return 'Kluster 6 (Elite)'
            elif row["Kluster"] == 7:
                return 'Kluster 7 (Premier)'
            elif row["Kluster"] == 8:
                return 'Kluster 8 (Prestige)'
            elif row["Kluster"] == 9:
                return 'Kluster 9 (Royal)'
            else:
                return 'Tidak Diketahui'

        RFM['group'] = RFM.apply(func, axis=1)
        st.header("Hasil Kluster")
        st.write(RFM)

        # Visualisasi Distribusi Kluster
        colors = ["DarkRed", "DarkCyan", "DarkBlue", "Yellow"][:num_clusters]
        result = RFM.group.value_counts()

        # Menampilkan Plot Hasil Klaster
        fig_result, ax_result = plt.subplots(figsize=(10, 6))
        result.plot(kind="barh", color=colors)
        ax_result.set_title("Result")
        ax_result.set_xlabel("Count")
        ax_result.set_ylabel("Group")
        st.header("Result")
        st.pyplot(fig_result)

        silhouette_avg = silhouette_score(scaled, RFM["Kluster"])
        st.write("silhouette score: ", silhouette_avg)

        st.write(result)

        # Medeskripsikan kluster sesuai yang di input.
        st.subheader("Deskripsi Kluster")
        if num_clusters == 2:
            st.write(
                """
            - **Kluster 1:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, memiliki frekuensi pembelian yang sedang, dan memiliki nilai moneter yang tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan dan meningkatkan keterlibatan pelanggan, serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.

            - **Kluster 2:**
                - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.
            """
            )
        elif num_clusters == 3:
            st.write(
                """
            - **Kluster 1:**
                - Pelanggan dalam kluster ini adalah pelanggan yang sangat aktif dan sering berbelanja dengan nilai moneter yang signifikan. Fokus strategi pemasaran dapat ditempatkan untuk mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

            - **Kluster 2:**
                - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.

            - **Kluster 3:**
                -  Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lebih lama dan memiliki nilai moneter yang signifikan. Strategi pemasaran dapat difokuskan untuk mempertahankan dan meningkatkan keterlibatan pelanggan serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.
                """
            )
        elif num_clusters == 4:
            st.write(
                """
            - **Kluster 1:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lebih lama, memiliki frekuensi pembelian yang sedang, dan memiliki nilai moneter yang signifikan. Strategi pemasaran dapat difokuskan pada mempertahankan dan meningkatkan keterlibatan pelanggan, serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.

            - **Kluster 2:**
                - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.

            - **Kluster 3:**
                - Pelanggan dalam kluster ini adalah pelanggan yang baru-baru ini berinteraksi dengan bisnis, sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

            - **Kluster 4:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.
                """
            )
        elif num_clusters == 5:
            st.write(
                """
            - **Kluster 1:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lebih lama, memiliki frekuensi pembelian yang sedang, dan memiliki nilai moneter yang tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan dan meningkatkan keterlibatan pelanggan, serta mendorong peningkatan nilai moneter melalui penawaran yang disesuaikan.

            - **Kluster 2:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sangat sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

            - **Kluster 3:**
                - Pelanggan dalam kluster ini baru-baru ini berinteraksi dengan bisnis tetapi memiliki frekuensi pembelian yang lebih rendah. Strategi pemasaran harus difokuskan pada meningkatkan frekuensi pembelian dan membangun keterlibatan pelanggan. Mungkin ada peluang untuk memahami lebih lanjut kebutuhan dan preferensi pelanggan ini untuk meningkatkan keterlibatan.

            - **Kluster 4:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sering berbelanja, dan memiliki nilai moneter yang tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.

            - **Kluster 5:**
                - Pelanggan dalam kluster ini adalah pelanggan dengan durasi interaksi yang lama, sangat sering berbelanja, dan memiliki nilai moneter yang sangat tinggi. Strategi pemasaran dapat difokuskan pada mempertahankan tingkat aktivitas yang tinggi dan mungkin meningkatkan nilai moneter dengan penawaran eksklusif atau program loyalitas yang ditargetkan.
                """
            )
        elif num_clusters == 6:
            st.write(
                """
            - **Kluster 1:**
                - Frekuensi Pembelian Rendah.
                - Nilai Moneter Tinggi.

            - **Kluster 2:**
                - Frekuensi Pembelian Sedang.
                - Nilai Moneter Tinggi.

            - **Kluster 3:**
                - Frekuensi Pembelian Tinggi.
                - Nilai Moneter Sangat Tinggi.

            - **Kluster 4:**
                - Frekuensi Pembelian Tinggi.
                - Nilai Moneter Tinggi.

            - **Kluster 5:**
                - Frekuensi Pembelian Rendah.
                - Nilai Moneter Rendah.

            - **Kluster 6:**
                - Frekuensi Pembelian Tinggi.
                - Nilai Moneter Sangat Tinggi.
            """
            )
        elif num_clusters == 7:
            st.write(
                """
            - **Kluster 1:**
                - Frekuensi Pembelian Rendah.
                - Nilai Moneter Tinggi.

            - **Kluster 2:**
                - Frekuensi Pembelian Tinggi.
                - Nilai Moneter Sangat Tinggi.

            - **Kluster 3:**
                - Frekuensi Pembelian Rendah.
                - Nilai Moneter Tinggi.

            - **Kluster 4:**
                - Frekuensi Pembelian Tinggi.
                - Nilai Moneter Tinggi.

            - **Kluster 5:**
                - Frekuensi Pembelian Sedang.
                - Nilai Moneter Tinggi.

            - **Kluster 6:**
                - Frekuensi Pembelian Tinggi.
                - Nilai Moneter Sangat Tinggi.

            - **Kluster 7:**
                - Frekuensi Pembelian Rendah.
                - Nilai Moneter Rendah.
            """
            )

        # Medeskripsikan kluster sesuai yang di input.
        st.subheader("Strategi Pemasaran")
        if num_clusters == 2:
            st.write(
                """
            1. **Kluster 1:**
            - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
            - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:** Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.

            2. **Kluster 2:**
            - Kampanye retargeting untuk meningkatkan keterlibatan.
            - Program insentif untuk mendorong peningkatan frekuensi pembelian.
            - **Kelemahan:** Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

            Penting untuk terus memantau dan mengevaluasi efektivitas strategi ini serta melakukan analisis lebih lanjut terkait dengan konteks bisnis dan tren pasar untuk memastikan relevansi strategi pemasaran seiring waktu.
            """
            )
        elif num_clusters == 3:
            st.write(
                """
            1. **Kluster 1:**
            - **Strategi Pemasaran:**
                - Program loyalitas eksklusif untuk meningkatkan retensi pelanggan.
                - Penawaran bundel atau diskon khusus untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

            2. **Kluster 2:**
            - **Strategi Pemasaran:**
                - Kampanye retargeting untuk meningkatkan keterlibatan.
                - Program insentif untuk mendorong peningkatan frekuensi pembelian.
            - **Kelemahan:**
                - Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

            3. **Kluster 3:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.
            """
            )
        elif num_clusters == 4:
            st.write(
                """
            1. **Kluster 1:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.

            2. **Kluster 2:**
            - **Strategi Pemasaran:**
                - Kampanye retargeting untuk meningkatkan keterlibatan.
                - Program insentif untuk mendorong peningkatan frekuensi pembelian.
            - **Kelemahan:**
                - Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

            3. **Kluster 3:**
            - **Strategi Pemasaran:**
                - Program loyalitas eksklusif untuk meningkatkan retensi pelanggan.
                - Penawaran bundel atau diskon khusus untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

            4. **Kluster 4:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

            Penting untuk terus memantau dan mengevaluasi efektivitas strategi ini serta melakukan analisis lebih lanjut terkait dengan konteks bisnis dan tren pasar untuk memastikan relevansi strategi pemasaran seiring waktu.
            """
            )
        elif num_clusters == 5:
            st.write(
                """
            1. **Kluster 1:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Frekuensi pembelian yang sedang dapat menunjukkan tantangan dalam meningkatkan aktivitas pembelian.

            2. **Kluster 2:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

            3. **Kluster 3:**
            - **Strategi Pemasaran:**
                - Kampanye retargeting untuk meningkatkan keterlibatan.
                - Program insentif untuk mendorong peningkatan frekuensi pembelian.
            - **Kelemahan:**
                - Tren frekuensi pembelian rendah dapat mengindikasikan risiko kehilangan minat.

            4. **Kluster 4:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

            5. **Kluster 5:**
            - **Strategi Pemasaran:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            - **Kelemahan:**
                - Risiko kelelahan pelanggan karena tingginya frekuensi pembelian.

            Penting untuk terus memantau dan mengevaluasi efektivitas strategi ini serta melakukan analisis lebih lanjut terkait dengan konteks bisnis dan tren pasar untuk memastikan relevansi strategi pemasaran seiring waktu.
            """
            )

        elif num_clusters == 6:
            st.write(
                """
            1. **Kluster 1:**
                - Kampanye retargeting untuk meningkatkan keterlibatan.
                - Program insentif untuk mendorong peningkatan frekuensi pembelian.

            2. **Kluster 2:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            3. **Kluster 3:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            4. **Kluster 4:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            5. **Kluster 5:**
                - Meningkatkan keterlibatan dengan kampanye personalisasi.
                - Program insentif untuk meningkatkan frekuensi pembelian.

            6. **Kluster 6:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.
            """
            )
        elif num_clusters == 7:
            st.write(
                """
            1. **Kluster 1:**
                - Kampanye retargeting untuk meningkatkan keterlibatan.
                - Program insentif untuk mendorong peningkatan frekuensi pembelian.

            2. **Kluster 2:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            3. **Kluster 3:**
                - Kampanye retargeting untuk meningkatkan keterlibatan.
                - Program insentif untuk mendorong peningkatan frekuensi pembelian.

            4. **Kluster 4:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            5. **Kluster 5:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            6. **Kluster 6:**
                - Program retensi pelanggan dengan fokus pada pengalaman pelanggan.
                - Pengembangan penawaran eksklusif untuk meningkatkan nilai moneter.

            7. **Kluster 7:**
                - Meningkatkan keterlibatan dengan kampanye personalisasi.
                - Program insentif untuk meningkatkan frekuensi pembelian.
            """
            )
elif selected == "Information":
    st.markdown(
        """
    <div style="text-align: center;">
        <h1>Segmentasi Customer</h1>
        <p>metode Klustering K-Means dengan model RFM</p>
    </div>
    """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Segmentasi Customer")
    st.write("Segmentasi Customer adalah proses mengelompokkan pelanggan ke dalam kategori-kategori atau segmen tertentu berdasarkan karakteristik atau perilaku yang serupa. ")
    st.subheader("Tujuan")
    st.write("Tujuan dari segmentasi Customer adalah untuk memahami kebutuhan, preferensi, dan perilaku pelanggan secara lebih mendalam, sehingga perusahaan dapat menyediakan layanan atau produk yang lebih sesuai dengan setiap kelompok pelanggan.")
    st.subheader("Model RFM")
    st.markdown("Segmentasi pelanggan menggunakan model RFM (Recency, Frequency, Monetary) adalah suatu pendekatan untuk mengelompokkan pelanggan berdasarkan perilaku pembelian mereka.  \nModel RFM ini mengevaluasi tiga dimensi penting dari aktivitas pelanggan:  \n  \n**-Recency (R):**  \nMenunjukkan seberapa baru pelanggan melakukan pembelian. Nilai Recency diukur dari waktu terakhir pelanggan melakukan pembelian. Semakin kecil nilai Recency, semakin baru pelanggan melakukan pembelian. \n\n - Rumus Recency = Tanggal Terakhir - Tanggal Pembelian Terakhir \n\n Keterangan : Tanggal Terakhir adalah tanggal terakhir data diambil.  \n  \n**-Frequency (F):**  \nMenunjukkan seberapa sering pelanggan melakukan pembelian. Nilai Frequency diukur dari jumlah total pembelian yang dilakukan oleh pelanggan. Semakin besar nilai Frequency, semakin sering pelanggan melakukan pembelian. \n\n - Rumus Frequency = Jumlah Pembelian Total \n\n Keterangan : Jumlah Pembelian Total adalah total jumlah pembelian yang dilakukan oleh pelanggan. \n  \n**-Monetary (M):**  \nMenunjukkan seberapa banyak uang yang dihabiskan oleh pelanggan. Nilai Monetary diukur dari total nilai pembelian yang dilakukan oleh pelanggan. Semakin besar nilai Monetary, semakin banyak uang yang dihabiskan pelanggan. \n \n - Rumus Monetary = Total Nilai Pembelian \n\nKeterangan : Total Nilai Pembelian adalah jumlah total uang yang dihabiskan oleh pelanggan. ")
    st.subheader("Normalisasi (Scaling) Data")
    st.write("Kode menggunakan objek StandardScaler dari library scikit-learn untuk melakukan normalisasi atau scaling terhadap data RFM.Proses normalisasi dilakukan untuk memastikan bahwa semua variabel dalam data memiliki skala yang seragam, sehingga perbedaan skala antar variabel tidak mempengaruhi hasil dari algoritma yang akan digunakan nantinya.Rumus umum normalisasi (z-score scaling) yang digunakan adalah:")
    st.latex(r'z = \frac{(x - \text{mean})}{\text{std}}')
    st.write("di mana x adalah nilai dalam dataset, mean adalah rata-rata dari dataset, dan std adalah deviasi standar dari dataset.")
    st.subheader("Elbow Method")
    st.write("Metode Elbow  adalah sebuah pendekatan yang digunakan untuk menentukan jumlah kluster optimal dalam algoritma K-Means atau metode clustering lainnya. Nama Elbow berasal dari bentuk grafik yang dihasilkan oleh nilai Inertia (within-cluster sum of squares) terhadap jumlah kluster. Inertia mengukur seberapa jauh titik-titik data dalam suatu kluster dari pusat kluster.")
    st.subheader("Visualisasi")
    st.write("Scatter Plot:")
    st.markdown(
        """
        - **Pengertian:** Scatter plot digunakan untuk memvisualisasikan distribusi data dalam ruang dua dimensi. Pada kode ini, scatter plot digunakan untuk menampilkan data pada dua fitur utama (Feature 1 dan Feature 2) dengan pewarnaan berdasarkan kluster hasil dari algoritma K-Means.
        - **Interpretasi:** Melalui scatter plot, Anda dapat melihat bagaimana data terpisah ke dalam kluster dan apakah kluster tersebut dapat dibedakan dengan jelas berdasarkan fitur yang diplot.
        """
    )
    st.write("Heatmap Korelasi:")
    st.markdown(
        """
        - **Pengertian:** Heatmap korelasi menunjukkan tingkat korelasi antara berbagai fitur pada dataset. Pada kode ini, heatmap korelasi digunakan untuk mengevaluasi korelasi antara variabel-variabel dalam dataset RFM.
        - **Interpretasi:** Heatmap korelasi membantu dalam memahami sejauh mana fitur-fitur dalam dataset berkorelasi. Ini dapat membantu mengidentifikasi hubungan yang kuat atau lemah antara fitur-fitur tersebut.
        """
    )
    st.write("Diagram Batang (Cluster Distribution):")
    st.markdown(
        """
        - **Pengertian:** Diagram batang digunakan untuk menunjukkan distribusi jumlah data di setiap kluster.
        - **Interpretasi:** Anda dapat melihat seberapa seimbang atau tidak seimbangnya distribusi data di antara kluster. Jika terdapat perbedaan signifikan dalam jumlah data di antara kluster, hal ini dapat memengaruhi hasil clustering.
        """
    )
    st.write("Box Plot of Recency by Cluster:")
    st.markdown(
        """
        - **Pengertian:** Box plot digunakan untuk menunjukkan distribusi dan statistik deskriptif dari variabel Recency di setiap kluster.
        - **Interpretasi:** Box plot membantu mengidentifikasi perbedaan distribusi nilai Recency di antara kluster. Outlier atau perbedaan signifikan dapat terlihat melalui box plot.
        """
    )
    st.write("Pair Plot:")
    st.markdown(
        """
        - **Pengertian:** Pair plot menunjukkan scatter plot pasangan dari variabel-variabel dalam dataset, dengan pewarnaan berdasarkan kluster.
        - **Interpretasi:** Pair plot memungkinkan Anda melihat hubungan simultan antara beberapa variabel dalam konteks kluster. Ini membantu dalam memahami pola dan perbedaan antar kluster.
        """
    )
    st.write("Violin Plot of Monetary by Cluster:")
    st.markdown(
        """
        - **Pengertian:** Violin plot digunakan untuk menunjukkan distribusi dan kepadatan data dari variabel Monetary di setiap kluster.
        - **Interpretasi:** Melalui violin plot, Anda dapat melihat seberapa bervariasi distribusi nilai Monetary di antara kluster. Puncak violin menunjukkan kepadatan data pada nilai tertentu.
        """
    )
    st.write("Standard Deviation (Inertia) Plot:")
    st.markdown(
        """
        - **Pengertian:** Plot standar deviasi (inertia) menunjukkan bagaimana nilai inertia berubah dengan jumlah kluster yang berbeda.
        - **Interpretasi:** Anda dapat mengidentifikasi jumlah kluster yang optimal dengan melihat titik di mana penurunan inertia mulai melambat. Nilai inertia yang lebih rendah menunjukkan bahwa kluster lebih kompak.
        """
    )
    st.write("Standard Deviation (Inertia) Value:")
    st.markdown(
        """
        - **Pengertian:** Nilai standar deviasi (inertia) di subplot ke-8 memberikan informasi tentang seberapa kompaknya kluster.
        - **Interpretasi:** Nilai ini memberikan gambaran tentang seberapa baik data di dalam kluster berkelompok. Nilai yang lebih rendah menunjukkan bahwa data di dalam kluster lebih seragam.
        """
    )
    st.subheader("Silhouette Score")
    st.write("Silhouette Score adalah metrik evaluasi yang digunakan untuk mengukur seberapa baik suatu objek telah diklasifikasikan dalam klusternya sendiri dibandingkan dengan kluster lainnya. Metrik ini memberikan nilai antara -1 dan 1, di mana nilai yang tinggi menunjukkan bahwa objek tersebut lebih baik cocok dengan kluster tempat ia berada daripada kluster tetangga terdekatnya.Formula Silhouette Score untuk setiap objek (i) dalam suatu kluster:")
    st.latex(r's(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}')
    st.write("di mana:")
    st.write("- \(s(i)\) adalah Silhouette Score untuk objek i.")
    st.write("- \(a(i)\) adalah rata-rata jarak antara objek i dan objek lain dalam kluster yang sama (intra-cluster distance).")
    st.write("- \(b(i)\) adalah rata-rata jarak antara objek i dan objek dalam kluster tetangga terdekat (inter-cluster distance).")
    st.write(
        "- \(\max\{a(i), b(i)\}\) adalah nilai maksimum antara \(a(i)\) dan \(b(i)\).")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Langkah-langkah:")
    st.markdown("**Pengolahan Data:**  \n---Membaca dataset dari file CSV dan mengganti nama kolom **kdplg** menjadi **ID Pelanggan**.  \n---Memfilter data transaksi pembelian antara tahun 2020 hingga 2023.")
    st.code("""
    # Baca Dataset dari CSV
    df = pd.read_csv("Data Sukun Jual.csv", sep=';')
    df.rename(columns={"kdplg": "ID Pelanggan"}, inplace=True)

    # Filter data berdasarkan tahun
    df_filtered = df[df['tanggal'].dt.year.between(2020, 2023)]
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "**Perhitungan RFM:**  \n---Menghitung nilai Recency, Frequency, dan Monetary untuk setiap pelanggan.")
    st.code("""
    # Recency (waktu terakhir pembelian).
    recency = df_filtered.groupby(["ID Pelanggan"]).agg(
        {"tanggal": lambda x: ((day - x.max()).days)})

    # Frequency (frekuensi pembelian).
    freq = df.drop_duplicates(subset="nota").groupby(
        ["ID Pelanggan"])[["nota"]].count()

    # Monetary (total nilai pembelian).
    df["total"] = df["jumlah"] * df["hgjual"]
    money = df.groupby(["ID Pelanggan"])[["total"]].sum()

    # Menggabungkan hasil Recency, Frequency, dan Monetary ke dalam DataFrame RFM.
    recency.columns = ["Recency"]
    freq.columns = ["Frequency"]
    money.columns = ["Monetary"]
    RFM = pd.concat([recency, freq, money], axis=1)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "**Normalisasi Data:**  \n---Menormalkan data RFM menggunakan StandardScaler.")
    st.code("""
    # Menormalisasi (scaling) data RFM sehingga memiliki skala yang seragam.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(RFM)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "**Pemilihan Jumlah Kluster Optimal:**  \n---Menggunakan metode Elbow untuk menentukan jumlah kluster optimal.")
    st.code("""
    # Menghitung Inertia untuk metode Elbow.
    inertia = []
    for i in np.arange(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=20)
        kmeans.fit(scaled)
        inertia.append(kmeans.inertia_)

    # Menampilkan Elbow Method Plot.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(1, 11), inertia, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    st.header("Elbow Method")
    st.pyplot(fig)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Interaktif Input dan Deskripsi Kluster:**  \n---Memberikan pengguna kemampuan untuk memasukkan jumlah kluster.  \n---Menampilkan deskripsi kluster berdasarkan jumlah kluster yang dimasukkan.")
    st.code("""
    # Meminta pengguna memasukkan jumlah kluster yang diinginkan.
    num_clusters = st.number_input(
        "Masukkan angka Kluster", min_value=1, max_value=10, value=4)

    # Menampilkan deskripsi kluster berdasarkan jumlah kluster yang dipilih.
    st.subheader("Deskripsi Kluster")
    if num_clusters == 1:
        st.write("Deskripsi Kluster 1")
    elif num_clusters == 2:
        st.write("Deskripsi Kluster 2")
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Penerapan K-Means dan Visualisasi Hasil Klustering:**  \n---Melakukan klustering menggunakan K-Means dengan jumlah kluster yang dipilih.  \n---Menampilkan visualisasi hasil klustering seperti scatter plot, heatmap, dan distribusi kluster.")
    st.code("""
    # Melakukan clustering menggunakan K-Means dengan jumlah kluster yang dipilih.
    kmeans = KMeans(n_clusters=num_clusters, random_state=20)
    kmeans.fit(scaled)
    RFM["Kluster"] = (kmeans.labels_ + 1)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Pengelompokkan dan Analisis:**  \n---Mengelompokkan data berdasarkan kluster.  \n---Menampilkan rata-rata nilai RFM untuk setiap kluster.")
    st.code("""
    # Menghitung rata-rata RFM untuk setiap kluster.
    final = RFM.groupby(["Kluster"])[
                        ["Recency", "Frequency", "Monetary"]].mean()
    st.header("Average RFM Values by Cluster")
    st.dataframe(final)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "**Visualisasi Distribusi Kluster:**  \n---Menampilkan distribusi jumlah pelanggan dalam setiap kluster.")
    st.code("""
    # Menampilkan distribusi kluster dalam bentuk bar chart.
    result = RFM.group.value_counts()
    fig_result, ax_result = plt.subplots(figsize=(10, 6))
    result.plot(kind="barh", color=colors)
    ax_result.set_title("Result")
    ax_result.set_xlabel("Count")
    ax_result.set_ylabel("Group")
    st.header("Result")
    st.pyplot(fig_result)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Evaluasi Kinerja Klustering:**  \n---Menggunakan metrik evaluasi seperti silhouette score, davies bouldin score, dan calinski harabasz score.")
    st.code("""
    # Menghitung metrik evaluasi klustering.
    silhouette_avg = silhouette_score(scaled, RFM["Kluster"])
    st.write("Silhouette Score: ", silhouette_avg)
    """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Interpretasi Hasil Klustering:**  \n---Memberikan interpretasi hasil klustering dengan memberikan label pada setiap kluster.")
    st.code("""
    # Interpretasi Hasil Clustering
    def func(row):
        if row["Kluster"] == 1:
            return 'Bronze'
        elif row["Kluster"] == 2:
            return 'Silver'
        elif row["Kluster"] == 3:
            return 'Gold'
        elif row["Kluster"] == 4:
            return 'Platinum'
        elif row["Kluster"] == 5:
            return 'Diamond'
        elif row["Kluster"] == 6:
            return 'Elite'
        elif row["Kluster"] == 7:
            return 'Premier'
        elif row["Kluster"] == 8:
            return 'Prestige'
        elif row["Kluster"] == 9:
            return 'Royal'
        elif row["Kluster"] == 10:
            return 'Imperial'
        elif row["Kluster"] == 11:
            return 'Ultimate'
        else:
            return 'Tidak Diketahui'

    # Menambahkan kolom 'group' berdasarkan hasil klustering.
    RFM['group'] = RFM.apply(func, axis=1)
    st.header("Hasil Kluster")
    st.write(RFM)
    """, language="python")
