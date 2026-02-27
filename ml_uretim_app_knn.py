# ============================================================
#   ÜRETİM ZEKÂ PLATFORMU — v2.0
#   Decision Tree + KNN: Regression & Classification
#
#   KURULUM:
#       pip install streamlit scikit-learn pandas numpy matplotlib seaborn
#
#   ÇALIŞTIRMA:
#       streamlit run ml_uretim_app_knn.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

# ── Sayfa Ayarları ──────────────────────────────────────────
st.set_page_config(
    page_title="Üretim Zekâ Platformu v2",
    page_icon="🏭",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 800; }
    .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 600; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ── VERİ SETLERİ
# ============================================================

@st.cache_data
def sentetik_regression_verisi():
    np.random.seed(42)
    n = 500
    uretim   = np.random.randint(50, 500, n)
    makine   = np.random.randint(1, 20, n)
    iscilik  = np.random.uniform(4, 16, n)
    hammadde = np.random.uniform(10, 80, n)
    vardiya  = np.random.choice([1, 2], n)
    maliyet  = (
        uretim * hammadde * 0.9
        + iscilik * 250
        + makine * 150
        + (vardiya == 2) * 1200
        + np.random.normal(0, 800, n)
    ).round(2)
    return pd.DataFrame({
        "uretim_miktari": uretim,
        "makine_yasi": makine,
        "iscilik_saati": iscilik.round(1),
        "hammadde_fiyati": hammadde.round(2),
        "vardiya": vardiya,
        "toplam_maliyet": maliyet,
    })


@st.cache_data
def ai4i_verisi_yukle(yuklenen=None):
    if yuklenen is not None:
        df = pd.read_csv(yuklenen)
    elif os.path.exists("ai4i2020.csv"):
        df = pd.read_csv("ai4i2020.csv")
    else:
        df = None

    if df is not None:
        df.columns = (df.columns.str.strip()
                      .str.replace(r"[\[\]()]", "", regex=True)
                      .str.replace(" ", "_").str.lower())
        if "type" in df.columns:
            le = LabelEncoder()
            df["type_encoded"] = le.fit_transform(df["type"])
        return df, False

    np.random.seed(42)
    n = 10000
    tip   = np.random.choice(["L","M","H"], n, p=[0.5,0.3,0.2])
    tip_e = np.where(tip=="L",0,np.where(tip=="M",1,2))
    air   = np.random.normal(300, 2, n)
    proc  = air + np.random.normal(10, 1, n)
    rpm   = np.random.normal(1538, 179, n).clip(1168, 2886)
    tork  = np.random.normal(40, 10, n).clip(3.8, 76.6)
    wear  = (tip_e*30 + np.random.uniform(0,200,n)).clip(0,250)
    power = tork * rpm * 2 * np.pi / 60
    failure = ((power > 9000) | (tork*(250-wear)>13000) |
               ((rpm<1380)&(tork>40)) | (np.random.rand(n)<0.001)).astype(int)
    df = pd.DataFrame({
        "type": tip, "type_encoded": tip_e,
        "air_temperature_k": air.round(1),
        "process_temperature_k": proc.round(1),
        "rotational_speed_rpm": rpm.round(0).astype(int),
        "torque_nm": tork.round(1),
        "tool_wear_min": wear.round(0).astype(int),
        "machine_failure": failure,
    })
    return df, True


# ============================================================
# ── MODEL EĞİTİM FONKSİYONLARI
# ============================================================

@st.cache_resource
def regression_modelleri_egit(df):
    X = df[["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]]
    y = df["toplam_maliyet"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Decision Tree ---
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(Xtr, ytr)
    dt_pred = dt.predict(Xte)

    # --- KNN (StandardScaler zorunlu!) ---
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(Xtr_s, ytr)
    knn_pred = knn.predict(Xte_s)

    dt_met = {
        "r2":   round(r2_score(yte, dt_pred), 4),
        "mae":  round(mean_absolute_error(yte, dt_pred), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, dt_pred))), 0),
    }
    knn_met = {
        "r2":   round(r2_score(yte, knn_pred), 4),
        "mae":  round(mean_absolute_error(yte, knn_pred), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, knn_pred))), 0),
    }
    return dt, knn, scaler, Xtr, Xte, ytr, yte, dt_pred, knn_pred, dt_met, knn_met


@st.cache_resource
def classification_modelleri_egit(df):
    ozellikler = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                               "rotational_speed_rpm","torque_nm","tool_wear_min"]
                  if c in df.columns]
    X = df[ozellikler]
    y = df["machine_failure"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Decision Tree ---
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(Xtr, ytr)
    dt_pred = dt.predict(Xte)

    # --- KNN ---
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr_s, ytr)
    knn_pred = knn.predict(Xte_s)

    def met(yte, pred):
        return {
            "accuracy":  round(accuracy_score(yte, pred)*100, 2),
            "precision": round(precision_score(yte, pred, zero_division=0)*100, 1),
            "recall":    round(recall_score(yte, pred, zero_division=0)*100, 1),
            "f1":        round(f1_score(yte, pred, zero_division=0)*100, 1),
        }

    return (dt, knn, scaler, Xtr, Xte, ytr, yte,
            dt_pred, knn_pred,
            met(yte, dt_pred), met(yte, knn_pred),
            ozellikler)


# ── K OPTİMİZASYON ANALİZİ ─────────────────────────────────
def k_optimizasyon_reg(Xtr, Xte, ytr, yte):
    """K=1..20 için eğitim/test R² döndürür."""
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    tr_list, te_list = [], []
    for k in range(1, 21):
        m = KNeighborsRegressor(n_neighbors=k).fit(Xtr_s, ytr)
        tr_list.append(r2_score(ytr, m.predict(Xtr_s)))
        te_list.append(r2_score(yte, m.predict(Xte_s)))
    return tr_list, te_list

def k_optimizasyon_clf(Xtr, Xte, ytr, yte):
    """K=1..20 için eğitim/test accuracy döndürür."""
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    tr_list, te_list = [], []
    for k in range(1, 21):
        m = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ytr)
        tr_list.append(accuracy_score(ytr, m.predict(Xtr_s)))
        te_list.append(accuracy_score(yte, m.predict(Xte_s)))
    return tr_list, te_list


# ============================================================
# ── VERİ YÜKLE
# ============================================================
reg_df = sentetik_regression_verisi()
ai4i_df, sentetik_mi = ai4i_verisi_yukle()

(dt_r, knn_r, scaler_r,
 Xtr_r, Xte_r, ytr_r, yte_r,
 dt_pred_r, knn_pred_r,
 dt_met_r, knn_met_r) = regression_modelleri_egit(reg_df)

(dt_c, knn_c, scaler_c,
 Xtr_c, Xte_c, ytr_c, yte_c,
 dt_pred_c, knn_pred_c,
 dt_met_c, knn_met_c,
 clf_ozellikler) = classification_modelleri_egit(ai4i_df)


# ============================================================
# ── BAŞLIK
# ============================================================
col_t1, col_t2 = st.columns([3,1])
with col_t1:
    st.title("🏭 Üretim Zekâ Platformu v2")
    st.markdown("**Decision Tree** ve **KNN** karşılaştırmalı makine öğrenmesi — Regression & Classification")
with col_t2:
    if sentetik_mi:
        st.warning("⚠️ Demo mod\nai4i2020.csv bulunamadı")
    else:
        st.success("✅ AI4I 2020\nGerçek veri yüklendi")

st.divider()

with st.expander("📂 AI4I 2020 CSV Yükle (opsiyonel)"):
    yuklenen = st.file_uploader("ai4i2020.csv", type=["csv"])
    if yuklenen:
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()


# ============================================================
# ── ANA SEKMELER
# ============================================================
ana1, ana2, ana3, ana4 = st.tabs([
    "📈 REGRESSION — Maliyet Tahmini",
    "🔴 CLASSIFICATION — Arıza Tahmini",
    "🆚 DT vs KNN Karşılaştırma",
    "🔍 K Optimizasyon Analizi",
])


# ════════════════════════════════════════════════════════════
# SEKME 1 — REGRESSION
# ════════════════════════════════════════════════════════════
with ana1:
    st.markdown("### 📈 Üretim Maliyeti Tahmini")
    st.markdown("*Sentetik üretim verisi (n=500)*")
    st.divider()

    # Model seçici
    model_sec_r = st.radio(
        "🤖 Model Seçin:",
        ["🌳 Decision Tree", "🔵 KNN (K=5)"],
        horizontal=True, key="model_r"
    )
    kullan_dt_r = "Decision Tree" in model_sec_r

    aktif_met_r  = dt_met_r  if kullan_dt_r else knn_met_r
    aktif_pred_r = dt_pred_r if kullan_dt_r else knn_pred_r

    st.divider()
    col_gir, col_sonu = st.columns([1, 2])

    with col_gir:
        st.markdown("#### ⚙️ Parametreler")
        r_uretim   = st.slider("📦 Üretim Miktarı (adet)", 50, 500, 250, key="r_ur")
        r_makine   = st.slider("🔧 Makine Yaşı (yıl)", 1, 20, 8, key="r_mk")
        r_iscilik  = st.slider("👷 İşçilik Saati", 4.0, 16.0, 10.0, step=0.5, key="r_is")
        r_hammadde = st.slider("🪨 Hammadde Fiyatı (TL)", 10, 80, 45, key="r_hm")
        r_vardiya  = st.radio("🌙 Vardiya", ["Gündüz", "Gece"], key="r_vr")
        r_vval     = 1 if r_vardiya == "Gündüz" else 2
        st.divider()
        r_btn = st.button("🔍 Maliyet Tahmin Et", type="primary", use_container_width=True, key="r_btn")

    with col_sonu:
        st.markdown(f"#### 📊 Model Performansı — {'Decision Tree' if kullan_dt_r else 'KNN'}")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Skoru", aktif_met_r["r2"])
        m2.metric("MAE", f"{aktif_met_r['mae']:,.0f} TL")
        m3.metric("RMSE", f"{aktif_met_r['rmse']:,.0f} TL")

        # DT vs KNN hızlı karşılaştırma
        delta_r2 = round(knn_met_r["r2"] - dt_met_r["r2"], 4)
        if kullan_dt_r:
            st.caption(f"KNN ile karşılaştırma: KNN R² = {knn_met_r['r2']} (fark: {delta_r2:+.4f})")
        else:
            st.caption(f"DT ile karşılaştırma: DT R² = {dt_met_r['r2']} (fark: {-delta_r2:+.4f})")

        st.divider()

        if r_btn:
            girdi = pd.DataFrame([{
                "uretim_miktari": r_uretim, "makine_yasi": r_makine,
                "iscilik_saati": r_iscilik, "hammadde_fiyati": r_hammadde,
                "vardiya": r_vval
            }])
            if kullan_dt_r:
                tahmin = dt_r.predict(girdi)[0]
            else:
                girdi_s = scaler_r.transform(girdi)
                tahmin  = knn_r.predict(girdi_s)[0]

            birim = tahmin / r_uretim
            c1, c2 = st.columns(2)
            c1.metric("💰 Tahmini Maliyet", f"{tahmin:,.0f} TL")
            c2.metric("📌 Birim Maliyet",   f"{birim:,.0f} TL/adet")

            if tahmin > 25000:
                st.error(f"⚠️ Yüksek maliyet — optimizasyon önerilir.")
            elif tahmin > 15000:
                st.warning(f"🟡 Orta maliyet seviyesi.")
            else:
                st.success(f"✅ Normal maliyet seviyesi.")

            # İki modeli karşılaştır
            tahmin_dt  = dt_r.predict(girdi)[0]
            tahmin_knn = knn_r.predict(scaler_r.transform(girdi))[0]
            st.info(f"🌳 DT tahmini: **{tahmin_dt:,.0f} TL** | 🔵 KNN tahmini: **{tahmin_knn:,.0f} TL** | Fark: {abs(tahmin_dt-tahmin_knn):,.0f} TL")
        else:
            st.info("← Parametreleri ayarlayıp **Maliyet Tahmin Et** butonuna tıklayın.")

    st.divider()

    # Grafikler
    gc1, gc2 = st.columns(2)

    with gc1:
        st.markdown("##### 🎯 Gerçek vs Tahmin")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(yte_r, dt_pred_r,  alpha=0.35, s=12, color="#e74c3c", label="Decision Tree")
        ax.scatter(yte_r, knn_pred_r, alpha=0.35, s=12, color="#3498db", label="KNN")
        mn, mx = float(yte_r.min()), float(yte_r.max())
        ax.plot([mn,mx],[mn,mx], "k--", lw=1.5, label="Mükemmel")
        ax.set_xlabel("Gerçek (TL)"); ax.set_ylabel("Tahmin (TL)")
        ax.set_title("DT vs KNN — Gerçek vs Tahmin", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig)

    with gc2:
        st.markdown("##### 📌 Özellik Önemi (Decision Tree)")
        imp = dt_r.feature_importances_
        fnames = ["Üretim\nMiktarı","Makine\nYaşı","İşçilik\nSaati","Hammadde\nFiyatı","Vardiya"]
        idx = np.argsort(imp)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        colors = ["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"]
        ax2.barh([fnames[i] for i in idx], imp[idx],
                  color=[colors[i] for i in idx], edgecolor="white")
        for i, v in enumerate(imp[idx]):
            ax2.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax2.set_xlabel("Önem Skoru")
        ax2.set_title("DT Özellik Önemi\n(KNN özellik önemi hesaplamaz)", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig2)

    # KNN notu
    st.info("ℹ️ **KNN Notu:** KNN'de özellik önemi hesaplanamaz — tüm özellikler eşit ağırlıkta mesafe hesabına girer. Bu nedenle özellik önemi yalnızca Decision Tree için gösterilmektedir.")

    # Overfitting — DT
    with st.expander("🔍 Overfitting Analizi — Decision Tree (max_depth)"):
        tr2_list, te2_list = [], []
        for d in range(1,16):
            m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(Xtr_r, ytr_r)
            tr2_list.append(r2_score(ytr_r, m.predict(Xtr_r)))
            te2_list.append(r2_score(yte_r, m.predict(Xte_r)))
        chart_df = pd.DataFrame({"Eğitim R²": tr2_list, "Test R²": te2_list}, index=range(1,16))
        chart_df.index.name = "max_depth"
        st.line_chart(chart_df, height=220)

    with st.expander("🌳 Karar Ağacı Kuralları (depth=3)"):
        sm = DecisionTreeRegressor(max_depth=3, random_state=42).fit(Xtr_r, ytr_r)
        st.code(export_text(sm, feature_names=list(Xtr_r.columns)), language="text")


# ════════════════════════════════════════════════════════════
# SEKME 2 — CLASSIFICATION
# ════════════════════════════════════════════════════════════
with ana2:
    st.markdown("### 🔴 Makine Arıza Tahmini (Classification)")
    veri_tipi = "Sentetik" if sentetik_mi else "AI4I 2020 Gerçek Veri"
    st.markdown(f"*{veri_tipi} (n={len(ai4i_df):,})*")
    st.divider()

    model_sec_c = st.radio(
        "🤖 Model Seçin:",
        ["🌳 Decision Tree", "🔵 KNN (K=5)"],
        horizontal=True, key="model_c"
    )
    kullan_dt_c = "Decision Tree" in model_sec_c
    aktif_met_c  = dt_met_c  if kullan_dt_c else knn_met_c
    aktif_pred_c = dt_pred_c if kullan_dt_c else knn_pred_c

    st.divider()

    col_gir2, col_sonu2 = st.columns([1, 2])

    with col_gir2:
        st.markdown("#### ⚙️ Parametreler")
        c_tip    = st.selectbox("🏷️ Ürün Tipi", ["L — Düşük","M — Orta","H — Yüksek"], key="c_tip")
        c_tipval = {"L":0,"M":1,"H":2}[c_tip[0]]
        c_air_c  = st.slider("🌡️ Hava Sıcaklığı (°C)", 22.0, 32.0, 27.0, step=0.1, key="c_air")
        c_air    = c_air_c + 273.15
        c_proc_c = st.slider("🔥 Proses Sıcaklığı (°C)", 32.0, 42.0, 37.0, step=0.1, key="c_proc")
        c_proc   = c_proc_c + 273.15
        c_rpm    = st.slider("⚡ Dönüş Hızı (rpm)", 1168, 2886, 1538, key="c_rpm")
        c_tork   = st.slider("🔩 Tork (Nm)", 3.8, 76.6, 40.0, step=0.1, key="c_tork")
        c_wear   = st.slider("🔧 Takım Aşınması (dk)", 0, 250, 100, key="c_wear")
        st.divider()
        c_btn = st.button("🔍 Arıza Riskini Tahmin Et", type="primary", use_container_width=True, key="c_btn")

    with col_sonu2:
        st.markdown(f"#### 📊 Model Performansı — {'Decision Tree' if kullan_dt_c else 'KNN'}")
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Accuracy",  f"%{aktif_met_c['accuracy']}")
        cm2.metric("Precision", f"%{aktif_met_c['precision']}")
        cm3.metric("Recall",    f"%{aktif_met_c['recall']}", help="Kritik metrik — arızaları yakalama oranı")
        cm4.metric("F1 Score",  f"%{aktif_met_c['f1']}")

        # Recall karşılaştırması
        st.caption(f"🌳 DT Recall: %{dt_met_c['recall']} | 🔵 KNN Recall: %{knn_met_c['recall']}")
        st.divider()

        if c_btn:
            girdi2_dict = {}
            for col in clf_ozellikler:
                if "type"    in col: girdi2_dict[col] = c_tipval
                elif "air"   in col: girdi2_dict[col] = c_air
                elif "process" in col: girdi2_dict[col] = c_proc
                elif "speed" in col or "rpm" in col: girdi2_dict[col] = c_rpm
                elif "torque" in col or "tork" in col: girdi2_dict[col] = c_tork
                elif "wear"  in col: girdi2_dict[col] = c_wear

            girdi2 = pd.DataFrame([girdi2_dict])

            if kullan_dt_c:
                pred = dt_c.predict(girdi2)[0]
                prob = dt_c.predict_proba(girdi2)[0]
            else:
                girdi2_s = scaler_c.transform(girdi2)
                pred = knn_c.predict(girdi2_s)[0]
                prob = knn_c.predict_proba(girdi2_s)[0]

            if pred == 1:
                st.error(f"🔴 **ARIZA RİSKİ YÜKSEK**\nArıza olasılığı: %{prob[1]*100:.1f}")
            else:
                st.success(f"✅ **NORMAL ÇALIŞMA**\nArıza olasılığı: %{prob[1]*100:.1f}")

            st.progress(float(prob[1]), text=f"Arıza olasılığı: %{prob[1]*100:.1f}")

            # Her iki modelin tahminini karşılaştır
            dt_pred_tek  = dt_c.predict(girdi2)[0]
            knn_pred_tek = knn_c.predict(scaler_c.transform(girdi2))[0]
            dt_prob_tek  = dt_c.predict_proba(girdi2)[0][1]
            knn_prob_tek = knn_c.predict_proba(scaler_c.transform(girdi2))[0][1]

            etiket_dt  = "🔴 ARIZA" if dt_pred_tek  == 1 else "✅ NORMAL"
            etiket_knn = "🔴 ARIZA" if knn_pred_tek == 1 else "✅ NORMAL"
            st.info(f"🌳 DT: **{etiket_dt}** (%{dt_prob_tek*100:.1f}) | 🔵 KNN: **{etiket_knn}** (%{knn_prob_tek*100:.1f})")
        else:
            st.info("← Makine parametrelerini girin ve tahmin butona basın.")

    st.divider()

    gc3, gc4 = st.columns(2)

    with gc3:
        st.markdown("##### 🗂️ Confusion Matrix — Decision Tree")
        cm_dt = confusion_matrix(yte_c, dt_pred_c)
        fig3, ax3 = plt.subplots(figsize=(4, 3.5))
        ax3.imshow(cm_dt, cmap="Reds")
        ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
        ax3.set_xticklabels(["Normal","Arıza"])
        ax3.set_yticklabels(["Normal","Arıza"])
        ax3.set_xlabel("Tahmin"); ax3.set_ylabel("Gerçek")
        ax3.set_title(f"DT — Recall: %{dt_met_c['recall']}", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, str(cm_dt[i,j]), ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if cm_dt[i,j] > cm_dt.max()/2 else "black")
        plt.tight_layout(); st.pyplot(fig3)

    with gc4:
        st.markdown("##### 🗂️ Confusion Matrix — KNN")
        cm_knn = confusion_matrix(yte_c, knn_pred_c)
        fig4, ax4 = plt.subplots(figsize=(4, 3.5))
        ax4.imshow(cm_knn, cmap="Blues")
        ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
        ax4.set_xticklabels(["Normal","Arıza"])
        ax4.set_yticklabels(["Normal","Arıza"])
        ax4.set_xlabel("Tahmin"); ax4.set_ylabel("Gerçek")
        ax4.set_title(f"KNN — Recall: %{knn_met_c['recall']}", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, str(cm_knn[i,j]), ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if cm_knn[i,j] > cm_knn.max()/2 else "black")
        plt.tight_layout(); st.pyplot(fig4)

    with st.expander("📋 Detaylı Sınıflandırma Raporu"):
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Decision Tree**")
            st.code(classification_report(yte_c, dt_pred_c, target_names=["Normal","Arıza"]))
        with r2:
            st.markdown("**KNN**")
            st.code(classification_report(yte_c, knn_pred_c, target_names=["Normal","Arıza"]))

    with st.expander("🌳 Karar Ağacı Görselleştirmesi (depth=3)"):
        sm2 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr_c, ytr_c)
        fig5, ax5 = plt.subplots(figsize=(16, 6))
        plot_tree(sm2, feature_names=clf_ozellikler,
                  class_names=["Normal","Arıza"],
                  filled=True, rounded=True, fontsize=8,
                  ax=ax5, impurity=False)
        ax5.set_title("Classification Tree (depth=3)", fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig5)


# ════════════════════════════════════════════════════════════
# SEKME 3 — KARŞILAŞTIRMA
# ════════════════════════════════════════════════════════════
with ana3:
    st.markdown("### 🆚 Decision Tree vs KNN — Detaylı Karşılaştırma")
    st.divider()

    # REGRESSION TABLOSU
    st.markdown("#### 📈 Regresyon Karşılaştırması")
    reg_tablo = pd.DataFrame({
        "Metrik":         ["R² Skoru", "MAE (TL)", "RMSE (TL)", "Ölçeklendirme Gerekir?",
                           "Özellik Önemi?", "Yorumlanabilirlik", "Hız (Tahmin)"],
        "🌳 Decision Tree": [dt_met_r["r2"], f"{dt_met_r['mae']:,.0f}", f"{dt_met_r['rmse']:,.0f}",
                            "❌ Hayır", "✅ Evet", "⭐⭐⭐ Yüksek", "⚡ Çok Hızlı"],
        "🔵 KNN":          [knn_met_r["r2"], f"{knn_met_r['mae']:,.0f}", f"{knn_met_r['rmse']:,.0f}",
                            "✅ Evet (StandardScaler)", "❌ Hayır", "⭐ Düşük", "🐢 Yavaş"],
    })
    st.dataframe(reg_tablo, hide_index=True, use_container_width=True)

    if dt_met_r["r2"] > knn_met_r["r2"]:
        st.success(f"✅ Regresyon kazananı: **Decision Tree** (R²: {dt_met_r['r2']} > {knn_met_r['r2']})")
    else:
        st.success(f"✅ Regresyon kazananı: **KNN** (R²: {knn_met_r['r2']} > {dt_met_r['r2']})")

    st.divider()

    # CLASSIFICATION TABLOSU
    st.markdown("#### 🔴 Sınıflandırma Karşılaştırması")
    clf_tablo = pd.DataFrame({
        "Metrik":         ["Accuracy", "Precision", "Recall ⭐", "F1 Score",
                           "Ölçeklendirme?", "Yorumlanabilirlik"],
        "🌳 Decision Tree": [f"%{dt_met_c['accuracy']}", f"%{dt_met_c['precision']}",
                            f"%{dt_met_c['recall']}", f"%{dt_met_c['f1']}",
                            "❌ Hayır", "⭐⭐⭐ Yüksek"],
        "🔵 KNN":          [f"%{knn_met_c['accuracy']}", f"%{knn_met_c['precision']}",
                            f"%{knn_met_c['recall']}", f"%{knn_met_c['f1']}",
                            "✅ Evet", "⭐ Düşük"],
    })
    st.dataframe(clf_tablo, hide_index=True, use_container_width=True)

    if dt_met_c["recall"] > knn_met_c["recall"]:
        st.success(f"✅ Recall kazananı: **Decision Tree** (%{dt_met_c['recall']} > %{knn_met_c['recall']})")
    else:
        st.success(f"✅ Recall kazananı: **KNN** (%{knn_met_c['recall']} > %{dt_met_c['recall']})")

    st.divider()

    # GENEL KARŞILAŞTIRMA TABLOSU
    st.markdown("#### 🔬 Algoritma Özellikleri")
    genel = pd.DataFrame({
        "Özellik": ["Algoritma Türü", "Eğitim Zamanı", "Tahmin Zamanı",
                    "Bellek Kullanımı", "Hiperparametre",
                    "Eksik Veri Toleransı", "Aykırı Değer Duyarlılığı",
                    "En İyi Olduğu Durum"],
        "🌳 Decision Tree": [
            "Kural tabanlı", "Hızlı O(n log n)", "Çok hızlı O(log n)",
            "Düşük (model kaydedilir)", "max_depth",
            "Orta", "Düşük duyarlılık",
            "Yorumlanabilirlik önemliyse"
        ],
        "🔵 KNN": [
            "Örnek tabanlı (lazy)", "Sıfır — model eğitilmez", "Yavaş O(n·d)",
            "Yüksek (tüm veri bellekte)", "K (komşu sayısı)",
            "Düşük", "Yüksek duyarlılık",
            "Yerel örüntüler önemliyse"
        ],
    })
    st.dataframe(genel, hide_index=True, use_container_width=True)

    st.warning("⚠️ **Kritik Fark:** KNN, tahmin sırasında tüm eğitim verisini bellekte tutar ve her tahmin için tüm veriyi tarar. AI4I'daki 10.000 kayıtta bu fark gözlemlenebilir; gerçek üretim ortamındaki milyonlarca kayıtta ise ciddi performans sorununa yol açar.")


# ════════════════════════════════════════════════════════════
# SEKME 4 — K OPTİMİZASYON
# ════════════════════════════════════════════════════════════
with ana4:
    st.markdown("### 🔍 KNN — K Değeri Optimizasyon Analizi")
    st.markdown("*Decision Tree'deki max_depth analizinin KNN karşılığı*")
    st.divider()

    opt1, opt2 = st.columns(2)

    with opt1:
        st.markdown("#### 📈 Regresyon — K vs R²")
        with st.spinner("K analizi hesaplanıyor..."):
            tr_r, te_r = k_optimizasyon_reg(Xtr_r, Xte_r, ytr_r, yte_r)

        best_k_r = int(np.argmax(te_r)) + 1
        fig_k1, ax_k1 = plt.subplots(figsize=(6, 4))
        ax_k1.plot(range(1,21), tr_r, "o-", color="#e74c3c", label="Eğitim R²", lw=2)
        ax_k1.plot(range(1,21), te_r, "s-", color="#3498db", label="Test R²", lw=2)
        ax_k1.axvline(best_k_r, color="green", linestyle="--", lw=1.5, label=f"En iyi K={best_k_r}")
        ax_k1.axvline(5, color="gray", linestyle=":", lw=1, label="Kullanılan K=5")
        ax_k1.set_xlabel("K (Komşu Sayısı)", fontsize=11)
        ax_k1.set_ylabel("R²", fontsize=11)
        ax_k1.set_title("K=1 → Overfitting\nBüyük K → Underfitting", fontsize=10, fontweight="bold")
        ax_k1.legend(fontsize=9); ax_k1.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_k1)

        st.success(f"🏆 Regresyon için en iyi K: **{best_k_r}** (Test R²: {te_r[best_k_r-1]:.4f})")
        st.dataframe(
            pd.DataFrame({"K": range(1,21), "Eğitim R²": [round(x,4) for x in tr_r],
                          "Test R²": [round(x,4) for x in te_r]}),
            hide_index=True, height=250
        )

    with opt2:
        st.markdown("#### 🔴 Sınıflandırma — K vs Accuracy")
        with st.spinner("K analizi hesaplanıyor..."):
            tr_c, te_c_k = k_optimizasyon_clf(Xtr_c, Xte_c, ytr_c, yte_c)

        best_k_c = int(np.argmax(te_c_k)) + 1
        fig_k2, ax_k2 = plt.subplots(figsize=(6, 4))
        ax_k2.plot(range(1,21), tr_c, "o-", color="#e74c3c", label="Eğitim Acc", lw=2)
        ax_k2.plot(range(1,21), te_c_k, "s-", color="#3498db", label="Test Acc", lw=2)
        ax_k2.axvline(best_k_c, color="green", linestyle="--", lw=1.5, label=f"En iyi K={best_k_c}")
        ax_k2.axvline(5, color="gray", linestyle=":", lw=1, label="Kullanılan K=5")
        ax_k2.set_xlabel("K (Komşu Sayısı)", fontsize=11)
        ax_k2.set_ylabel("Accuracy", fontsize=11)
        ax_k2.set_title("K=1 → Overfitting\nBüyük K → Underfitting", fontsize=10, fontweight="bold")
        ax_k2.legend(fontsize=9); ax_k2.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_k2)

        st.success(f"🏆 Sınıflandırma için en iyi K: **{best_k_c}** (Test Acc: {te_c_k[best_k_c-1]*100:.2f}%)")
        st.dataframe(
            pd.DataFrame({"K": range(1,21),
                          "Eğitim Acc": [f"%{x*100:.2f}" for x in tr_c],
                          "Test Acc":   [f"%{x*100:.2f}" for x in te_c_k]}),
            hide_index=True, height=250
        )

    st.divider()
    st.markdown("#### 📚 K Seçimi Yorumu")
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        st.markdown("""
**K=1 — Tam Overfitting:**
Her nokta yalnızca kendine en yakın komşuya bakıyor.
Eğitim hatası sıfır, test hatası yüksek.

**K çok büyük (K→n) — Underfitting:**
Tüm veri kümesinin ortalamasına bakıyor.
Model hiçbir şey öğrenemiyor.
        """)
    with col_y2:
        st.markdown("""
**Optimal K — Eğitim/Test dengesi:**
Grafikte eğitim ve test eğrilerinin
en az uzaklaştığı bölge optimum K'yı verir.

**Karar Ağacı ile analoji:**
- K küçük = max_depth büyük (overfitting)
- K büyük = max_depth küçük (underfitting)
        """)

    st.divider()
    st.caption("📚 Veri: Sentetik üretim verisi (Regression) | AI4I 2020 Predictive Maintenance Dataset — Matzka (2020) (Classification)")
