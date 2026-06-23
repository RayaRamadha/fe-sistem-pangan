"""
train_all_models.py
===================
Melatih model KNN per kombinasi Komoditas × Provinsi dan menyimpannya
ke folder models/ sebagai {komoditas}_{provinsi}.pkl

Letakkan script ini satu folder dengan:
  - all_dataset_pangan.csv
  - app.py
  - last_known_prices.json  (sudah digenerate oleh generate_last_prices.py)

Cara pakai:
  python train_all_models.py

Setelah selesai, jalankan Flask:
  python app.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from datetime import datetime, timedelta

import yfinance as yf
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── CONFIG (harus sama dengan app.py) ────────────────────────
CSV_PATH      = "all_dataset_pangan.csv"
MODELS_DIR    = "models"
OIL_LAG_DAYS  = 14
LAG_WINDOWS   = [1, 3, 7, 14]
HISTORY_SIZE  = 20
MIN_ROWS      = 30          # minimum baris per kombinasi untuk dilatih
KNN_K         = 5           # jumlah tetangga KNN

FEATURE_COLS = [f'lag_{l}' for l in LAG_WINDOWS] + [
    'day_of_year', 'month', 'week', 'day_of_week',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'oil_price', 'Oil_Lag_14'
]

os.makedirs(MODELS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

# ── Provinsi mapping (sama dengan generate_last_prices.py) ───
def get_provinsi(wilayah):
    w = str(wilayah).upper().strip()
    prov_map = {
        'Aceh': ['ACEH','BENER MERIAH','BIREUEN','GAYO LUES','NAGAN RAYA','PIDIE','SIMEULUE','LANGSA','LHOKSEUMAWE','SABANG','SUBULUSSALAM','BANDA ACEH'],
        'Sumatera Utara': ['ASAHAN','DAIRI','DELI SERDANG','HUMBANG HASUNDUTAN','KARO','LANGKAT','MANDAILING NATAL','NIAS','PAKPAK BHARAT','SAMOSIR','SERDANG BEDAGAI','SIMALUNGUN','TAPANULI','TOBA','PEMATANGSIANTAR','TANJUNG BALAI','BATU BARA','LABUHANBATU','PADANG LAWAS','BINJAI','GUNUNGSITOLI','MEDAN','PADANG SIDEMPUAN','SIBOLGA','TEBING TINGGI'],
        'Sumatera Barat': ['AGAM','DHARMASRAYA','MENTAWAI','LIMA PULUH KOTA','PASAMAN','PESISIR SELATAN','SIJUNJUNG','TANAH DATAR','SOLOK','PADANG PARIAMAN','BUKITTINGGI','PADANG PANJANG','PARIAMAN','PAYAKUMBUH','SAWAHLUNTO','KOTA PADANG'],
        'Riau': ['BENGKALIS','INDRAGIRI','KAMPAR','MERANTI','KUANTAN SINGINGI','PELALAWAN','ROKAN','SIAK','DUMAI','PEKANBARU'],
        'Kepulauan Riau': ['BINTAN','KARIMUN','ANAMBAS','LINGGA','NATUNA','TANJUNG PINANG','BATAM'],
        'Jambi': ['BATANGHARI','BUNGO','KERINCI','MERANGIN','SAROLANGUN','TANJUNG JABUNG','TEBO','MUARO JAMBI','JAMBI','SUNGAI PENUH'],
        'Sumatera Selatan': ['BANYUASIN','EMPAT LAWANG','LAHAT','MUARA ENIM','MUSI','OGAN','PENUKAL ABAB','LUBUK LINGGAU','PAGAR ALAM','PALEMBANG','PRABUMULIH'],
        'Bengkulu': ['BENGKULU','KAUR','KEPAHIANG','LEBONG','MUKO MUKO','MUKOMUKO','REJANG LEBONG','SELUMA'],
        'Lampung': ['LAMPUNG','MESUJI','PESAWARAN','PESISIR BARAT','PRINGSEWU','TULANG BAWANG','TANGGAMUS','WAY KANAN','METRO','BANDAR LAMPUNG'],
        'Kepulauan Bangka Belitung': ['BANGKA','BELITUNG','PANGKAL PINANG'],
        'DKI Jakarta': ['JAKARTA','KEP. SERIBU','KEPULAUAN SERIBU'],
        'Jawa Barat': ['BANDUNG','BEKASI','BOGOR','CIREBON','INDRAMAYU','PANGANDARAN','PURWAKARTA','SUKABUMI','TASIKMALAYA','CIAMIS','CIANJUR','GARUT','KARAWANG','KUNINGAN','MAJALENGKA','SUBANG','SUMEDANG','CIMAHI','DEPOK'],
        'Jawa Tengah': ['BANJARNEGARA','BANYUMAS','BATANG','BLORA','BOYOLALI','BREBES','CILACAP','DEMAK','GROBOGAN','JEPARA','KARANGANYAR','KEBUMEN','KENDAL','KLATEN','KUDUS','MAGELANG','PATI','PEKALONGAN','PEMALANG','PURBALINGGA','PURWOREJO','REMBANG','SEMARANG','SRAGEN','SUKOHARJO','TEGAL','TEMANGGUNG','WONOGIRI','WONOSOBO','SALATIGA','SURAKARTA','SOLO'],
        'DI Yogyakarta': ['BANTUL','GUNUNGKIDUL','KULON PROGO','SLEMAN','YOGYAKARTA'],
        'Jawa Timur': ['BANGKALAN','BANYUWANGI','BLITAR','BOJONEGORO','BONDOWOSO','GRESIK','JEMBER','JOMBANG','KEDIRI','LAMONGAN','LUMAJANG','MADIUN','MAGETAN','MALANG','MOJOKERTO','NGANJUK','NGAWI','PACITAN','PAMEKASAN','PASURUAN','PONOROGO','PROBOLINGGO','SAMPANG','SIDOARJO','SITUBONDO','SUMENEP','TRENGGALEK','TUBAN','TULUNGAGUNG','BATU','SURABAYA'],
        'Banten': ['SERANG','TANGERANG','CILEGON','LEBAK','PANDEGLANG'],
        'Bali': ['BADUNG','BANGLI','BULELENG','GIANYAR','JEMBRANA','KARANGASEM','KLUNGKUNG','TABANAN','DENPASAR'],
        'Nusa Tenggara Barat': ['BIMA','DOMPU','LOMBOK','SUMBAWA','MATARAM'],
        'Nusa Tenggara Timur': ['ALOR','BELU','ENDE','FLORES','KUPANG','LEMBATA','MALAKA','MANGGARAI','NGADA','NAGEKEO','ROTE NDAO','SABU RAIJUA','SIKKA','SUMBA','TIMOR'],
        'Kalimantan Barat': ['BENGKAYANG','KAPUAS','KAYONG UTARA','KETAPANG','LANDAK','MELAWI','MEMPAWAH','SAMBAS','SANGGAU','SEKADAU','SINTANG','KUBU RAYA','PONTIANAK','SINGKAWANG'],
        'Kalimantan Tengah': ['BARITO','GUNUNG MAS','KAPUAS','KATINGAN','KOTAWARINGIN','LAMANDAU','MURUNG RAYA','PULANG PISAU','SUKAMARA','SERUYAN','PALANGKARAYA'],
        'Kalimantan Selatan': ['BALANGAN','BANJAR','BARITO KUALA','HULU SUNGAI','KOTABARU','TABALONG','TANAH BUMBU','TANAH LAUT','TAPIN','BANJARBARU','BANJARMASIN'],
        'Kalimantan Timur': ['BERAU','KUTAI','MAHAKAM ULU','PASER','PENAJAM','BALIKPAPAN','BONTANG','SAMARINDA'],
        'Kalimantan Utara': ['BULUNGAN','MALINAU','NUNUKAN','TANA TIDUNG','TARAKAN'],
        'Sulawesi Utara': ['BOLAANG','SANGIHE','SIAU TAGULANDANG','TALAUD','MINAHASA','BITUNG','KOTAMOBAGU','TOMOHON','MANADO'],
        'Sulawesi Tengah': ['BANGGAI','BUOL','DONGGALA','MOROWALI','PARIGI MOUTONG','POSO','SIGI','TOJO UNA UNA','TOJO UNA-UNA','TOLI TOLI','TOLI-TOLI','PALU'],
        'Sulawesi Selatan': ['BANTAENG','BARRU','BONE','BULUKUMBA','ENREKANG','GOWA','JENEPONTO','SELAYAR','LUWU','MAROS','PANGKAJENE','PINRANG','SIDENRENG RAPPANG','SINJAI','SOPPENG','TAKALAR','TANA TORAJA','TORAJA UTARA','WAJO','PARE PARE','PAREPARE','MAKASSAR','PALOPO'],
        'Sulawesi Tenggara': ['BOMBANA','BUTON','KOLAKA','KONAWE','MUNA','WAKATOBI','BAU BAU','BAU-BAU','KENDARI'],
        'Gorontalo': ['BOALEMO','BONE BOLANGO','POHUWATO','GORONTALO'],
        'Sulawesi Barat': ['MAJENE','MAMASA','MAMUJU','POLEWALI MANDAR','PASANGKAYU'],
        'Maluku': ['BURU','KEPULAUAN ARU','MALUKU','SERAM BAGIAN','TANIMBAR','AMBON','TUAL'],
        'Maluku Utara': ['HALMAHERA','SULA','MOROTAI','TALIABU','TERNATE','TIDORE'],
        'Papua': ['ASMAT','BIAK NUMFOR','BOVEN DIGOEL','DEIYAI','DOGIYAI','INTAN JAYA','JAYAPURA','JAYAWIJAYA','KEEROM','YAPEN','LANNY JAYA','MAMBERAMO','MAPPI','MERAUKE','MIMIKA','NABIRE','NDUGA','PANIAI','PUNCAK','SARMI','SUPIORI','TOLIKARA','WAROPEN','YAHUKIMO','YALIMO'],
        'Papua Barat': ['FAKFAK','FAK FAK','KAIMANA','MANOKWARI','MAYBRAT','ARFAK','RAJA AMPAT','SORONG','TAMBRAUW','BINTUNI','WONDAMA'],
    }
    for prov, keywords in prov_map.items():
        if any(key in w for key in keywords):
            return prov
    return 'Lainnya'


# ── Fetch oil data via yfinance ───────────────────────────────
def fetch_oil_data(start='2023-01-01'):
    print("🛢️  Mengambil data harga minyak dari yfinance...")
    end = datetime.now().strftime('%Y-%m-%d')
    oil_raw = yf.download('BZ=F', start=start, end=end, progress=False)
    if isinstance(oil_raw.columns, pd.MultiIndex):
        oil_raw.columns = oil_raw.columns.get_level_values(0)
    oil = oil_raw[['Close']].rename(columns={'Close': 'oil_price'})
    oil = oil.resample('D').ffill()
    oil[f'Oil_Lag_{OIL_LAG_DAYS}'] = oil['oil_price'].shift(OIL_LAG_DAYS)
    oil = oil.reset_index()
    oil.rename(columns={oil.columns[0]: 'ds'}, inplace=True)
    oil['ds'] = pd.to_datetime(oil['ds']).astype('datetime64[us]')
    oil.dropna(inplace=True)
    oil.reset_index(drop=True, inplace=True)
    print(f"   ✅ {len(oil)} baris oil data ({oil['ds'].min().date()} – {oil['ds'].max().date()})")
    return oil


# ── Time features ─────────────────────────────────────────────
def add_time_features(df):
    df = df.copy()
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month']       = df['Date'].dt.month
    df['week']        = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['Date'].dt.weekday
    df['month_sin']   = np.sin(2 * np.pi * df['month']       / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month']       / 12)
    df['dow_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']     = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


# ── Build lag features for a single series ────────────────────
def build_features(group_df, oil_df):
    """
    group_df: DataFrame dengan kolom ['Date', 'Price'] sudah di-sort ascending.
    Kembalikan DataFrame siap latih dengan FEATURE_COLS + 'target'.
    """
    df = group_df[['Date', 'Price']].copy().sort_values('Date').reset_index(drop=True)
    df['Date'] = df['Date'].astype('datetime64[us]')

    # Lag harga
    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Price'].shift(lag)

    # Time features
    df = add_time_features(df)

    # Merge oil data (left join ke tanggal terdekat ≤ Date)
    oil = oil_df[['ds', 'oil_price', f'Oil_Lag_{OIL_LAG_DAYS}']].copy()
    oil = oil.rename(columns={'ds': 'Date'})
    df = pd.merge_asof(
        df.sort_values('Date'),
        oil.sort_values('Date'),
        on='Date',
        direction='backward'
    )

    df['target'] = df['Price']
    df.dropna(inplace=True)
    return df


# ── Train single model ────────────────────────────────────────
def train_model(feat_df):
    X = feat_df[FEATURE_COLS].values
    y = feat_df['target'].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    knn = KNeighborsRegressor(n_neighbors=min(KNN_K, len(X_sc) - 1), metric='euclidean')

    # Eval on held-out test set
    metrics = None
    if len(X_sc) >= 20:
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.15, shuffle=False)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)
        mae  = float(mean_absolute_error(y_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        # MAPE — hindari division by zero
        mask = y_te != 0
        mape = float(np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100) if mask.any() else None
        metrics = {'mae': round(mae, 2), 'rmse': round(rmse, 2), 'mape': round(mape, 2) if mape else None}

    # Retrain pakai semua data
    knn.fit(X_sc, y)

    return {
        'knn':          knn,
        'scaler':       scaler,
        'feature_cols': FEATURE_COLS,
    }, metrics


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # 1. Baca & bersihkan dataset
    print("📂 Membaca dataset...")
    df = pd.read_csv(CSV_PATH)
    df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price', 'Wilayah', 'Komoditas'])
    df = df[~df['Wilayah'].str.contains('PERIODE|Sumber Data|Wilayah', na=False, case=False)]
    print(f"   Total baris valid: {len(df):,}")

    # 2. Mapping provinsi
    print("🗺️  Mapping wilayah → provinsi...")
    df['Provinsi'] = df['Wilayah'].apply(get_provinsi)
    df = df[df['Provinsi'] != 'Lainnya']

    # 3. Agregasi harian
    print("📊 Agregasi harian per Komoditas × Provinsi...")
    df_agg = (
        df.groupby(['Date', 'Komoditas', 'Provinsi'], as_index=False)['Price']
        .median()
        .sort_values(['Komoditas', 'Provinsi', 'Date'])
    )
    combinations = df_agg.groupby(['Komoditas', 'Provinsi'])
    total = len(combinations)
    print(f"   {total} kombinasi ditemukan")

    # 4. Ambil oil data
    # Gunakan tanggal paling awal di dataset sebagai start
    oil_start = (df_agg['Date'].min() - timedelta(days=OIL_LAG_DAYS + 10)).strftime('%Y-%m-%d')
    oil_df = fetch_oil_data(start=oil_start)

    # 5. Loop training
    print(f"\n🚀 Mulai training {total} model...\n")
    trained    = 0
    skipped    = 0
    errors     = 0
    mae_list   = []
    all_metrics = {}

    for idx, ((komoditas, provinsi), group) in enumerate(combinations, 1):
        key      = f"{komoditas}_{provinsi}"
        safe_key = key.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(MODELS_DIR, f"{safe_key}.pkl")

        prefix = f"[{idx:>4}/{total}] {komoditas[:30]:<30} | {provinsi:<22}"

        if len(group) < MIN_ROWS:
            print(f"{prefix} ⚠️  SKIP  (hanya {len(group)} baris)")
            skipped += 1
            continue

        try:
            feat_df = build_features(group, oil_df)

            if len(feat_df) < MIN_ROWS:
                print(f"{prefix} ⚠️  SKIP  (setelah lag: {len(feat_df)} baris)")
                skipped += 1
                continue

            bundle, metrics = train_model(feat_df)
            joblib.dump(bundle, out_path)
            trained += 1

            if metrics:
                mae_str = f"MAE Rp {metrics['mae']:,.0f} | RMSE Rp {metrics['rmse']:,.0f} | MAPE {metrics['mape']:.1f}%" if metrics['mape'] else f"MAE Rp {metrics['mae']:,.0f}"
                mae_list.append(metrics['mae'])
                all_metrics[safe_key] = {
                    'komoditas': komoditas,
                    'provinsi': provinsi,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'n_samples': len(feat_df),
                }
            else:
                mae_str = "MAE -"
            print(f"{prefix} ✅ OK   {mae_str}  ({len(feat_df)} baris)")

        except Exception as e:
            print(f"{prefix} ❌ ERROR: {e}")
            errors += 1

    # 6. Simpan metrics ke file JSON
    with open(METRICS_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Metrics disimpan ke: {METRICS_FILE}")

    # 7. Ringkasan
    print("\n" + "═" * 70)
    print(f"✅ Trained  : {trained}")
    print(f"⚠️  Skipped  : {skipped}")
    print(f"❌ Errors   : {errors}")
    if mae_list:
        print(f"📉 Rata-rata MAE : Rp {np.mean(mae_list):,.0f}")
        print(f"   Median MAE   : Rp {np.median(mae_list):,.0f}")
    print(f"\n📁 Model tersimpan di: ./{MODELS_DIR}/")
    print("   Jalankan 'python app.py' untuk memulai server.")


if __name__ == '__main__':
    main()