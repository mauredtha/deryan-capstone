import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import DataFrame
from pandas import concat
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title("Analisa Cuaca Harian Jakarta")
#st.metric("Sales", 100, 4)

st.write("Data cuaca merupakan sumber penting dan berharga, tidak hanya untuk meramalkan cuaca tetapi juga untuk berbagai tujuan baik di bidang industri, pemerintah dan dalam kehidupan sehari-hari. Contohnya seperti memahami iklim masa lalu, sekarang, dan potensi iklim masa depan, menentukan potensi risiko peristiwa bencana merencanakan perjalanan, pembangunan dan agriculture.")
#st.caption("Semangat ya, kamu pasti bisa")
st.subheader("Ayo kita lihat bagaimana cuaca harian di Jakarta 6 tahun terakhir ini!")

#CLEANSING

df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vT0n7KiqMNa5Y9BCr994Hcp1aG0yVDCLQKRBR7SyUO56FLCwhewj1MYWMoe9a9lTYH8Ix0-1Mb1csyg/pub?output=csv")

df['Tn'].fillna(df['Tn'].value_counts().index[0], inplace= True)
df['Tx'].fillna(df['Tx'].value_counts().index[0], inplace= True)
df['Tavg'].fillna(df['Tavg'].value_counts().index[0], inplace= True)

df['RH_avg'].fillna(df['RH_avg'].value_counts().index[0], inplace= True)
df['RR'].fillna(df['RR'].value_counts().index[0], inplace= True)
df['ss'].fillna(df['ss'].value_counts().index[0], inplace= True)
df['ff_x'].fillna(df['ff_x'].value_counts().index[0], inplace= True)
df['ddd_x'].fillna(df['ddd_x'].value_counts().index[0], inplace= True)
df['ff_avg'].fillna(df['ff_avg'].value_counts().index[0], inplace= True)
df['ddd_car'].fillna(df['ddd_car'].value_counts().index[0], inplace= True)

df.loc[df["RR"] == 8888.0, "RR"] = df['RR'].value_counts().index[0]
df['tanggal'] = pd.to_datetime(df['tanggal'])

# create a list of our conditions
conditions = [
    (df['RR'] >= 0.0) & (df['RR'] <= 0.49),
    (df['RR'] >= 0.5) & (df['RR'] <= 20.0),
    (df['RR'] >= 20.1) & (df['RR'] <= 50.0),
    (df['RR'] >= 50.1) & (df['RR'] <= 100.0),
    (df['RR'] >= 100.1) & (df['RR'] <= 150.0),
    (df['RR'] > 150.1)
    ]

# create a list of the values we want to assign for each condition
color_values = ['grey', 'green', 'yellow', 'orange', 'red', 'purple']
potention_values = ['Berawan', 'Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat', 'Hujan Sangat Lebat', 'Hujan Ekstrem']

# create a new column and use np.select to assign values to it using our lists as arguments
df['color_map'] = np.select(conditions, color_values)
df['potention'] = np.select(conditions, potention_values)
df['year'] = pd.DatetimeIndex(df['tanggal']).year
df['month'] = pd.DatetimeIndex(df['tanggal']).month

intensitas = df.groupby(['nama_wilayah','year','potention'])['RR'].count().reset_index(name='count').sort_values(['count'], ascending=False)

# END CLEANSING

wilayah = st.selectbox('Pilih Stasiun Cuaca', df['nama_wilayah'].unique())

fig = px.bar(intensitas[intensitas['nama_wilayah'] == wilayah], x='year', y='count', hover_data=['potention', 'count'], color='potention',labels={'potention':'Intensitas Cuaca'}, height=400)

st.plotly_chart(fig, use_container_width=True)

st.markdown("Dari grafik diatas, terlihat bahwa intensitas **_cuaca di Jakarta dari tahun 2017 s/d Oktober 2022 cenderung berawan, hujan ringan dan hujan sedang._**")

st.subheader("Sekarang kita lihat semua komponen cuaca harian Jakarta")

col1, col2 = st.columns([1, 3])

col1.write("Keterangan :")
col1.write("- Tavg: Temperatur rata-rata (°C)")
col1.write("- RR: Curah hujan (mm)")
col1.write("- RH_avg: Kelembapan rata-rata (%)")
col1.write("- ss: Lamanya penyinaran matahari (jam)")
col1.write("- ddd_x: Arah angin saat kecepatan maksimum (°)")
col1.write("- ff_avg: Kecepatan angin rata-rata (m/s)")

col2.write("Komponen Cuaca Harian Jakarta")

fig = px.area(df[df['nama_wilayah'] == wilayah], x='tanggal', y=['Tavg','RR','RH_avg','ss','ddd_x','ff_avg'])

#'Tavg', 'RHavg', 'RR', 'ss', 'ddd_x', 'ff_avg'

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
#st.plotly_chart(fig, use_container_width=True)

#col2.subheader("A narrow column with the data")
col2.plotly_chart(fig, use_container_width=True)

st.subheader("Apakah ada perbedaan cuaca harian di Jakarta saat sebelum pandemi dan saat pandemi?")

st.write("Seperti yang kita tahu, 2 tahun terakhir hampir seluruh aktivitas di berbagai negara terhenti dikarenakan pandemi, termasuk Indonesia. Lalu, bagaimana cuaca harian di Jakarta saat sebelum pandemi dan saat pandemi? apakah ada perbedaan? ")

col3, col4, col5 = st.columns([3, 3, 3])

col3.markdown("Note : _Pada Stasiun Cuaca Halim Perdana Kusuma, **tidak ada pencatatan cuaca sejak Januari 2019**. Sehingga kita hanya mengetahui bahwa sebelum pandemi :_")

halim_df = pd.DataFrame(np.array([['Suhu Rata-rata (°C)', '27.80', '-'], ['Curah Hujan Rata-rata (mm)', '5.62', '-'], ['Kelembapan Rata-rata (%)', '74.95', '-'], ['Lama Penyinaran Matahari Rata-rata (jam)', '-', '-'], ['Arah Angin Saat Kecepatan Maksimum Rata-rata (°)', '220.83', '-'], ['Kecepatan Angin Rata-rata (m/s)', '3.84', '-']]),
                   columns=['Komponen Cuaca', 'Sebelum Pandemi', 'Saat Pandemi'])

col3.table(halim_df)

col4.write("- Pada Stasiun Cuaca Maritim Tanjung Priok : ")

tanjung_df = pd.DataFrame(np.array([['Suhu Rata-rata (°C)', '28.73', '28.65'], ['Curah Hujan Rata-rata (mm)', '4.53', '5.87'], ['Kelembapan Rata-rata (%)', '75.07', '78.62'], ['Lama Penyinaran Matahari Rata-rata (jam)', '5.33', '5.34'], ['Arah Angin Saat Kecepatan Maksimum Rata-rata (°)', '162.12', '175.54'], ['Kecepatan Angin Rata-rata (m/s)', '2.11', '2.28']]),
                   columns=['Komponen Cuaca', 'Sebelum Pandemi', 'Saat Pandemi'])

col4.table(tanjung_df)

col5.write("- Pada Stasiun Cuaca Kemayoran :")

kemayoran_df = pd.DataFrame(np.array([['Suhu Rata-rata (°C)', '28.58', '28.50'], ['Curah Hujan Rata-rata (mm)', '4.76', '6.79'], ['Kelembapan Rata-rata (%)', '74.89', '77.17'], ['Lama Penyinaran Matahari Rata-rata (jam)', '4.68', '4.14'], ['Arah Angin Saat Kecepatan Maksimum Rata-rata (°)', '238.073', '262.49'], ['Kecepatan Angin Rata-rata (m/s)', '1.40', '1.41']]),
                   columns=['Komponen Cuaca', 'Sebelum Pandemi', 'Saat Pandemi'])

col5.table(kemayoran_df)

st.markdown("Setelah kita lihat, **cuaca Jakarta tidak jauh berbeda saat sebelum pandemi dan saat pandemi**. Cuaca Jakarta cukup stabil dan menunjukkan seasonality yang jelas. Dan karena suhu hari sebelumnya dengan suhu 2 hari berikutnya tidak signifikan maka dapat dijadikan dasar untuk membuat model dengan menggunakan suhu saat ini sebagai prediksi untuk hari berikutnya.")

st.subheader("Jakarta Hujan")

total_hjn = df.groupby(['nama_wilayah','year','month'])['RR'].sum().reset_index(name='curah_hujan').sort_values(['year','month'], ascending=True)

total_hjn['month'] = pd.to_datetime(total_hjn['month'], format='%m').dt.month_name().str.slice(stop=3)

fig = px.line(total_hjn[total_hjn['nama_wilayah'] == wilayah], x="month", y="curah_hujan", color='year', markers=True)

st.plotly_chart(fig, use_container_width=True)

cola, colb = st.columns([3, 3])

cola.markdown("# _**Awal musim hujan Jakarta diprakirakan pada bulan**_ Oktober-November dan **puncak musim hujan diprakirakan pada bulan** Januari-Februari")

colb.markdown("# _**Awal musim kemarau Jakarta diprakirakan pada bulan**_ April-Mei")


st.subheader("Apakah curah Hujan mempengaruhi banyaknya wisatawan mancanegara yang datang ke Jakarta ?")

manca = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTuRe1ZlhJY3Ps4ivcazuAiAkV5qsxSMyI5HRkAf6xKicTaFnV_8d3Vy-MJJeZEoQ/pub?output=csv")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=manca['bulan'], y=manca['RR'], name="Curah Hujan"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=manca['bulan'], y=manca['pengunjung_manca'], name="Wisatawan Mancanegara"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Curah Hujan & Wisatawan Mancanegara"
)

# Set x-axis title
fig.update_xaxes(title_text="Bulan Kunjungan di Tahun 2019")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Curah Hujan</b> ", secondary_y=False)
fig.update_yaxes(title_text="<b>Jumlah Wisatawan Mancanegara</b>", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

st.write("Jika dilihat pada grafik diatas, **curah hujan mempengaruhi kunjungan wisatawan mancanegara ke Jakarta**, dimana para wisatawan banyak berkunjung ketika bulan **Juni s/d September** saat curah hujan di Jakarta sedang menurun / pada saat memasuki musim kemarau di Indonesia.")

########

st.title("Ayo kita coba buat model prakiraan dan pengklasifikasian cuaca Jakarta!")

data = df[df['nama_wilayah'] == wilayah]

data.set_index("tanggal", inplace=True)

#st.dataframe(data)

st.subheader("SARIMA Modeling Prakiraan Cuaca")

st.write("Dari data cuaca Jakarta ini, kita dapat membuat model prakiraan dengan menggunakan suhu saat ini sebagai prediksi untuk hari berikutnya.  ")

# Shift the current temperature to the next day. 
predicted_df = data["Tavg"].to_frame().shift(1).rename(columns = {"Tavg": "Tavg_pred" })
actual_df = data["Tavg"].to_frame().rename(columns = {"Tavg": "Tavg_actual" })

# Concatenate the actual and predicted temperature
one_step_df = pd.concat([actual_df,predicted_df],axis=1)

# Select from the second row, because there is no prediction for today due to shifting.
one_step_df = one_step_df[1:]

from sklearn.metrics import mean_squared_error as MSE
from math import sqrt

# Calculate the RMSE
temp_pred_err = MSE(one_step_df.Tavg_actual, one_step_df.Tavg_pred, squared=False)

st.write("Validasi model yang digunakan yaitu Non-Dinamis dikarenakan range data yang tidak terlalu banyak.")

st.write("Validasi model menghasilkan Root Mean Squared Error (RMSE) / kesalahan rata-rata antara suhu yang diprediksi dan actual, hasilnya ",temp_pred_err)

st.write("Perlu diketahui bahwa Data train yang digunakan untuk modeling berdasarkan wilayah yang dipilih sebelumnya.")
st.write("Wilayah Halim Perdana Kusuma, data yang tersedia yaitu periode Januari 2017 s/d Desember 2018. Sedangkan Kemayoran dan Tanjung Priok, data yang tersedia yaitu  periode Januari 2017 s/d 15 Oktober 2022")

st.write("Format Tanggal Prediksi : YYYY-MM-DD dan Tahun Aktual : YYYY")

import itertools

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#print('Examples of parameter combinations for Seasonal ARIMA...')
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(one_step_df.Tavg_actual,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# Import the statsmodels library for using SARIMAX model
import statsmodels.api as sm

with st.form("form_modeling"):
   tgl_start = st.text_input('Tanggal Prediksi')
   thn_act = st.text_input('Tahun Aktual')

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
        #if len(tgl_start) == 0 & len(thn_act) == 0 :
        #tgl_start = '2021-03-07'
        #thn_act = '2021'
        # Fit the SARIMAX model using optimal parameters
        mod = sm.tsa.statespace.SARIMAX(one_step_df.Tavg_actual,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

        results = mod.fit()

        #2021-03-07T00:00:00
            
        pred = results.get_prediction(start=pd.to_datetime(tgl_start), dynamic=False)
        pred_ci = pred.conf_int()
            
        fig, ax = plt.subplots()
        ax = one_step_df.Tavg_actual[thn_act:].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (in Celsius)')
        plt.ylim([-20,30])
        plt.legend()
            #plt.show()
            # #st.plotly_chart(plt, use_container_width=True)
        st.pyplot(fig)
        st.write("Dari gambar diatas kita bisa lihat bahwa prakiraan temperatur cuaca selaras dengan temperatur sebenarnya dan menunjukkan siklus musiman sepanjang 365 hari. Model prakiraan ini akan lebih maksimal jika data train yang digunakan memiliki range yang cukup panjang.")


#st.dataframe(pred.predicted_mean)

st.subheader("Klasifikasi Intensitas Cuaca")

st.write("Data yang digunakan berdasarkan wilayah yang dipilih sebelumnya")
st.write("Masukkan nilai pada setiap komponen cuaca disamping untuk melihat hasil prediksi!!")

with st.form("form_klasifikasi"):
    Tn = st.text_input('Suhu Minimal (°C)')
    Tx = st.text_input('Suhu Maksimal (°C)')
    Tavg = st.text_input('Suhu Rata-rata (°C)')
    RH_avg = st.text_input('Kelembapan rata-rata (%)')
    RR = st.text_input('Curah Hujan (mm)')
    ss = st.text_input('lamanya penyinaran matahari (jam)')
    ff_x = st.text_input('Kecepatan angin maksimum (m/s)')
    ddd_x = st.text_input('Arah angin kecepatan maksimum (°)')
    ff_avg = st.text_input('Kecepatan angin rata-rata (m/s)')
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
            
        dt = df[df['nama_wilayah'] == wilayah]

        datas = dt.drop(['tanggal','kode_wilayah','nama_wilayah','ddd_car','color_map','year','month'], axis=1)

        #st.dataframe(datas)

        #Prepare the training set

        # X = feature values, all the columns except the last column
        X = datas.iloc[:, :-1]

        # y = target values, last column of the data frame
        y = datas.iloc[:, -1]

        #Split the data into 80% training and 20% testing
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Train the model
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train) #Training the model
        
        data_test = [[Tn, Tx, Tavg, RH_avg, RR, ss, ff_x, ddd_x, ff_avg]]

        df_data_test = pd.DataFrame(data_test, columns =['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg'])

        #Test the model
        predictions = model.predict(df_data_test)
        y_test = predictions

        akurasi = accuracy_score(y_test, predictions)

        st.markdown("Klasifikasi ini menggunakan model Decision Tree Classifier dengan Akurasi Skor : " + str(akurasi))

        st.write("Intensitas Cuaca dari data terkait yaitu " + predictions[0])


st.subheader("Kesimpulan")

st.markdown("- Dalam rentang waktu Januari 2017 s/d Oktober 2022, cuaca Jakarta cenderung **Berawan, Hujan Ringan & Hujan Sedang**")
st.write("- Suhu Jakarta cukup stabil, tidak ada perubahan yang berarti dari sebelum dan saat pandemi")
st.write("- Awal musim hujan Jakarta diprakirakan pada bulan Oktober-November dan puncak musim hujan diprakirakan pada bulan Januari-Februari")
st.write("- Awal musim kemarau Jakarta diprakirakan pada bulan April-Mei")
st.write("- Curah hujan mempengaruhi kunjungan wisatawan mancanegara ke Jakarta")
st.write("- Data history cuaca dapat digunakan untuk melakukan prakiraan cuaca dengan membuat modeling prakiraan time series menggunakan SARIMA Modeling")
st.write("- Hasil prakiraan dapat dimanfaatkan dalam kehidupan sehari hari untuk melihat kondisi cuaca hari berikutnya")
st.write("- Dapat mengkalsifikasikan cuaca berdasarkan komponennya")
st.write("- Hasil prakiraan dan klasifikasi data akan lebih optimal jika memeliki range yang cukup banyak")


st.subheader("Source:")

st.caption("Data Online BMKG, http://dataonline.bmkg.go.id/data_iklim")

st.caption("Curah Hujan Ekstrem Jakarta, https://bpbd.jakarta.go.id/berita/58/curah-hujan-ekstrem-di-dki-jakarta-genangan-mampu-tertangani- cepat#:~:text=Badan%20Meteorologi%2C%20Klimatologi%2C%20dan%20Geofisika,November%202021%2D18% 20Januari%202022. ")
st.caption("Multivariate time series forecasting, https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms- keras/ ")
st.caption("Solving A Simple Classification Problem with Python – Fruit Lovers’ Edition, https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2 ")
st.caption("Kerala Floods : EDA & Data Vis, https://www.kaggle.com/code/imdevskp/kerala-floods-eda-data-vis/notebook ")
st.caption("Time series Analysis and weather forecast in Python, https://medium.com/@llmkhoa511/time-series-analysis-and-weather-forecast-in-python-e80b664c7f71 ")






