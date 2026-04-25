import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, warnings, io
warnings.filterwarnings("ignore")
 
st.set_page_config(page_title="Üretim Zekâ Platformu", page_icon="🏭", layout="wide")
 
# ── VERİ ÜRETİMİ ─────────────────────────────────────────────
@st.cache_data
def sentetik_regression_verisi():
    np.random.seed(42)
    n = 500
    uretim   = np.random.randint(50, 500, n)
    makine   = np.random.randint(1, 20, n)
    iscilik  = np.round(np.random.uniform(4, 16, n), 1)
    hammadde = np.round(np.random.uniform(10, 80, n), 1)
    vardiya  = np.random.choice([1, 2], n)
    gurultu  = np.random.normal(0, 500, n)
    maliyet  = (uretim * hammadde * 0.9 + iscilik * 250 +
                makine * 150 + (vardiya == 2) * 1200 + gurultu)
    return pd.DataFrame({
        "uretim_miktari": uretim, "makine_yasi": makine,
        "iscilik_saati": iscilik, "hammadde_fiyati": hammadde,
        "vardiya": vardiya, "toplam_maliyet": np.round(maliyet, 0)
    })
 
def _parse_ai4i(df):
    """Sütun adlarını normalize et ve type_encoded ekle."""
    df.columns = (df.columns.str.strip()
                  .str.replace(r"[\[\]()]", "", regex=True)
                  .str.replace(" ", "_").str.lower())
    if "type" in df.columns:
        le = LabelEncoder()
        df["type_encoded"] = le.fit_transform(df["type"])
    return df
 
@st.cache_data(show_spinner=False)
def ai4i_url_indir():
    """UCI'dan AI4I 2020 verisini indir. Başarısız olursa None döner."""
    try:
        import requests
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            return _parse_ai4i(df)
    except Exception:
        pass
    return None
 
def ai4i_verisi_yukle(yuklenen_bytes=None):
    """Öncelik sırası: 1) yüklenen dosya  2) UCI URL  3) sentetik."""
    # 1 — Kullanıcı dosya yüklediyse
    if yuklenen_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(yuklenen_bytes))
        return _parse_ai4i(df), False
 
    # 2 — Yerel dosya (geliştirme ortamı)
    if os.path.exists("ai4i2020.csv"):
        return _parse_ai4i(pd.read_csv("ai4i2020.csv")), False
 
    # 3 — UCI'dan otomatik indir
    df = ai4i_url_indir()
    if df is not None:
        return df, False
 
    # 4 — Sentetik fallback
    np.random.seed(42)
    n = 10000
    tip   = np.random.choice(["L","M","H"], n, p=[0.5,0.3,0.2])
    tip_e = np.where(tip=="L",0,np.where(tip=="M",1,2))
    air   = np.random.normal(300,2,n)
    proc  = air + np.random.normal(10,1,n)
    rpm   = np.random.normal(1538,179,n).clip(1168,2886)
    tork  = np.random.normal(40,10,n).clip(3.8,76.6)
    wear  = (tip_e*30 + np.random.uniform(0,200,n)).clip(0,250)
    power = tork * rpm * 2 * np.pi / 60
    failure = ((power>9000) | (tork*(250-wear)>13000) |
               ((rpm<1380)&(tork>40)) | (np.random.rand(n)<0.001)).astype(int)
    return pd.DataFrame({
        "type":tip, "type_encoded":tip_e,
        "air_temperature_k":air.round(1),
        "process_temperature_k":proc.round(1),
        "rotational_speed_rpm":rpm.round(0).astype(int),
        "torque_nm":tork.round(1),
        "tool_wear_min":wear.round(0).astype(int),
        "machine_failure":failure
    }), True
 
# ── MODEL EĞİTİMİ ─────────────────────────────────────────────
@st.cache_data
def regresyon_egit(df_json):
    df = pd.read_json(io.StringIO(df_json))
    X = df[["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]]
    y = df["toplam_maliyet"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
 
    dt = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr, ytr)
 
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
 
    knn = KNeighborsRegressor(n_neighbors=5).fit(Xtr_s, ytr)
    svr = SVR(kernel="rbf", C=100, epsilon=0.1).fit(Xtr_s, ytr)
 
    dt_pred  = dt.predict(Xte)
    knn_pred = knn.predict(Xte_s)
    svr_pred = svr.predict(Xte_s)
 
    def met(pred):
        return {"r2": round(r2_score(yte,pred),4),
                "mae": round(mean_absolute_error(yte,pred),0),
                "rmse": round(float(np.sqrt(mean_squared_error(yte,pred))),0)}
 
    feat_imp = list(dt.feature_importances_)
    col_names = list(Xtr.columns)
 
    return ({"dt":met(dt_pred), "knn":met(knn_pred), "svr":met(svr_pred)},
            Xtr.to_json(), Xte.to_json(),
            ytr.to_json(), yte.to_json(),
            list(dt_pred), list(knn_pred), list(svr_pred),
            scaler.mean_.tolist(), scaler.scale_.tolist(),
            feat_imp, col_names)
 
@st.cache_data
def siniflandirma_egit(df_json, _kaynak=None):
    df = pd.read_json(io.StringIO(df_json))
    cols = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                        "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
    X = df[cols]
    y = df["machine_failure"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
 
    dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr, ytr)
 
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
 
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ytr)
    svc = SVC(kernel="rbf", C=1.0, probability=True, random_state=42).fit(Xtr_s, ytr)
 
    dt_pred  = dt.predict(Xte)
    knn_pred = knn.predict(Xte_s)
    svc_pred = svc.predict(Xte_s)
 
    def met(pred):
        return {"acc":  round(accuracy_score(yte,pred)*100,2),
                "prec": round(precision_score(yte,pred,zero_division=0)*100,1),
                "rec":  round(recall_score(yte,pred,zero_division=0)*100,1),
                "f1":   round(f1_score(yte,pred,zero_division=0)*100,1)}
 
    return ({"dt":met(dt_pred), "knn":met(knn_pred), "svc":met(svc_pred)},
            Xtr.to_json(), Xte.to_json(),
            ytr.to_json(), yte.to_json(),
            list(dt_pred), list(knn_pred), list(svc_pred),
            scaler.mean_.tolist(), scaler.scale_.tolist(),
            cols)
 
# ── VERİ VE MODEL YÜKLEME ─────────────────────────────────────
with st.spinner("Modeller eğitiliyor..."):
    reg_df = sentetik_regression_verisi()
 
    # AI4I — önce dosya yükleyici kontrol et
    yuklenen_bytes = st.session_state.get("ai4i_bytes", None)
    with st.spinner("AI4I 2020 verisi yükleniyor..."):
        ai4i_df, sentetik_mi = ai4i_verisi_yukle(yuklenen_bytes)
 
    (r_met, Xtr_r_j, Xte_r_j, ytr_r_j, yte_r_j,
     dt_pred_r, knn_pred_r, svr_pred_r,
     r_scaler_mean, r_scaler_scale,
     feat_imp, feat_names) = regresyon_egit(reg_df.to_json())
 
    (c_met, Xtr_c_j, Xte_c_j, ytr_c_j, yte_c_j,
     dt_pred_c, knn_pred_c, svc_pred_c,
     c_scaler_mean, c_scaler_scale,
     clf_cols) = siniflandirma_egit(ai4i_df.to_json(), sentetik_mi)
 
# Numpy dizilerine çevir
yte_r = np.array(pd.read_json(io.StringIO(yte_r_j)).squeeze())
yte_c = np.array(pd.read_json(io.StringIO(yte_c_j)).squeeze())
dt_pred_r  = np.array(dt_pred_r)
knn_pred_r = np.array(knn_pred_r)
svr_pred_r = np.array(svr_pred_r)
dt_pred_c  = np.array(dt_pred_c)
knn_pred_c = np.array(knn_pred_c)
svc_pred_c = np.array(svc_pred_c)
r_scaler_mean  = np.array(r_scaler_mean)
r_scaler_scale = np.array(r_scaler_scale)
c_scaler_mean  = np.array(c_scaler_mean)
c_scaler_scale = np.array(c_scaler_scale)
 
def r_scale(x): return (np.array(x) - r_scaler_mean) / r_scaler_scale
def c_scale(x): return (np.array(x) - c_scaler_mean) / c_scaler_scale
 
def dt_reg_predict(vals):
    # Yeniden eğit (hafif, cached değil ama hızlı)
    X = reg_df[feat_names]; y = reg_df["toplam_maliyet"]
    Xtr,_,ytr,_ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr, ytr)
    return m.predict(pd.DataFrame([dict(zip(feat_names, vals))]))[0]
 
# ── BAŞLIK ────────────────────────────────────────────────────
st.title("🏭 Üretim Zekâ Platformu")
st.markdown("**Decision Tree · KNN · SVM** — Regresyon ve Sınıflandırma karşılaştırması")
 
if not sentetik_mi:
    st.success("✅ AI4I 2020 gerçek verisi yüklendi — 10.000 gözlem, %3,4 arıza")
else:
    st.warning("⚠️ AI4I 2020 verisi otomatik indirilemedi. Lütfen aşağıdan yükleyin.")
    with st.expander("📂 AI4I 2020 CSV Yükle", expanded=True):
        st.markdown("Veriyi [UCI sayfasından](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) indirip yükleyebilirsiniz.")
        uploaded = st.file_uploader("ai4i2020.csv dosyasını seçin", type=["csv"])
        if uploaded:
            st.session_state["ai4i_bytes"] = uploaded.read()
            st.rerun()
 
st.divider()
 
# ── SEKMELER ─────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📈 Maliyet Tahmini",
    "🔴 Arıza Tahmini",
    "🆚 Model Karşılaştırma",
    "🔍 Hiperparametre Analizi",
    "📚 Algoritma Teorisi"
])
 
# ════════════════════════════════════════════════════════════
# SEKME 1 — REGRESYON / MALİYET TAHMİNİ
# ════════════════════════════════════════════════════════════
with t1:
    st.markdown("### 📈 Üretim Maliyeti Tahmini")
    st.markdown("Sol taraftaki parametreleri ayarlayın — tahmin **anında** güncellenir.")
    st.divider()
 
    col_gir, col_sonu = st.columns([1, 2])
 
    with col_gir:
        st.markdown("#### ⚙️ Üretim Parametreleri")
        r_uretim   = st.slider("📦 Üretim Miktarı (adet)", 50, 500, 250)
        r_makine   = st.slider("🔧 Makine Yaşı (yıl)", 1, 20, 8)
        r_iscilik  = st.slider("👷 İşçilik Saati", 4.0, 16.0, 10.0, step=0.5)
        r_hammadde = st.slider("🪨 Hammadde Fiyatı (TL/birim)", 10, 80, 45)
        r_vardiya  = st.radio("🌙 Vardiya", ["Gündüz", "Gece"])
        r_vval     = 1 if r_vardiya == "Gündüz" else 2
        st.divider()
        model_r = st.radio("🤖 Aktif Model", ["Decision Tree", "KNN", "SVM"], horizontal=True)
 
    with col_sonu:
        # Tahmin — anlık, buton gerektirmez
        girdi_ham = np.array([[r_uretim, r_makine, r_iscilik, r_hammadde, r_vval]], dtype=float)
        girdi_scaled = r_scale(girdi_ham[0]).reshape(1, -1)
 
        # Basit regresyon formülü ile DT tahmini (önbellek sorunlarını önler)
        tahmin_dt  = (r_uretim * r_hammadde * 0.9 + r_iscilik * 250 +
                      r_makine * 150 + (r_vval == 2) * 1200)
        # KNN: ağırlıklı ortalama (yaklaşık, hızlı)
        X_all = reg_df[feat_names].values
        y_all = reg_df["toplam_maliyet"].values
        X_all_s = (X_all - r_scaler_mean) / r_scaler_scale
        dists = np.sqrt(((X_all_s - girdi_scaled) ** 2).sum(axis=1))
        k_idx = np.argsort(dists)[:5]
        tahmin_knn = y_all[k_idx].mean()
        # SVM: linear approximation
        tahmin_svm = tahmin_dt * 1.02  # fallback — gerçek model aşağıda
 
        # Gerçek model tahmini (sklearn objeleri cache'den alınamıyor, yeniden eğit)
        @st.cache_data
        def predict_all_reg(ur, mk, is_, hm, vr):
            df = sentetik_regression_verisi()
            X = df[["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]]
            y = df["toplam_maliyet"]
            Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            sc = StandardScaler().fit(Xtr)
            girdi = pd.DataFrame([{"uretim_miktari":ur,"makine_yasi":mk,
                                    "iscilik_saati":is_,"hammadde_fiyati":hm,"vardiya":vr}])
            girdi_s = sc.transform(girdi)
            dt  = DecisionTreeRegressor(max_depth=5,random_state=42).fit(Xtr,ytr)
            knn = KNeighborsRegressor(n_neighbors=5).fit(sc.transform(Xtr),ytr)
            svr = SVR(kernel="rbf",C=100,epsilon=0.1).fit(sc.transform(Xtr),ytr)
            return float(dt.predict(girdi)[0]), float(knn.predict(girdi_s)[0]), float(svr.predict(girdi_s)[0])
 
        tahmin_dt, tahmin_knn, tahmin_svm = predict_all_reg(
            r_uretim, r_makine, r_iscilik, r_hammadde, r_vval)
 
        if   model_r == "Decision Tree": tahmin = tahmin_dt
        elif model_r == "KNN":           tahmin = tahmin_knn
        else:                            tahmin = tahmin_svm
 
        birim = tahmin / r_uretim
 
        st.markdown(f"#### 💰 Tahmin Sonucu — {model_r}")
        c1, c2 = st.columns(2)
        c1.metric("Toplam Maliyet", f"{tahmin:,.0f} TL")
        c2.metric("Birim Maliyet",  f"{birim:,.0f} TL/adet")
 
        if tahmin > 25000:
            st.error("⚠️ Yüksek maliyet — optimizasyon önerilir.")
        elif tahmin > 15000:
            st.warning("🟡 Orta maliyet seviyesi.")
        else:
            st.success("✅ Normal maliyet seviyesi.")
 
        st.info(f"🌳 DT: **{tahmin_dt:,.0f} TL** &nbsp;|&nbsp; 🔵 KNN: **{tahmin_knn:,.0f} TL** &nbsp;|&nbsp; 🟣 SVM: **{tahmin_svm:,.0f} TL**")
 
        st.divider()
        st.markdown("#### 📊 Test Seti Performansı")
        perf = pd.DataFrame({
            "Model": ["🌳 Decision Tree", "🔵 KNN", "🟣 SVM"],
            "R²":    [r_met["dt"]["r2"], r_met["knn"]["r2"], r_met["svr"]["r2"]],
            "MAE":   [f"{r_met['dt']['mae']:,.0f} TL", f"{r_met['knn']['mae']:,.0f} TL", f"{r_met['svr']['mae']:,.0f} TL"],
            "RMSE":  [f"{r_met['dt']['rmse']:,.0f} TL", f"{r_met['knn']['rmse']:,.0f} TL", f"{r_met['svr']['rmse']:,.0f} TL"],
        })
        st.dataframe(perf, hide_index=True, use_container_width=True)
 
    st.divider()
 
    # Grafikler
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### 🎯 Gerçek vs Tahmin (Test Seti)")
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.scatter(yte_r, dt_pred_r,  alpha=0.3, s=10, color="#e74c3c", label="DT")
        ax.scatter(yte_r, knn_pred_r, alpha=0.3, s=10, color="#3498db", label="KNN")
        ax.scatter(yte_r, svr_pred_r, alpha=0.3, s=10, color="#9b59b6", label="SVM")
        mn, mx = float(yte_r.min()), float(yte_r.max())
        ax.plot([mn,mx],[mn,mx],"k--",lw=1.5,label="Mükemmel")
        ax.set_xlabel("Gerçek (TL)"); ax.set_ylabel("Tahmin (TL)")
        ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig)
 
    with g2:
        st.markdown("##### 📌 Özellik Önemi — Decision Tree")
        labels = ["Üretim\nMiktarı","Makine\nYaşı","İşçilik\nSaati","Hammadde\nFiyatı","Vardiya"]
        imp = np.array(feat_imp)
        idx = np.argsort(imp)
        fig2, ax2 = plt.subplots(figsize=(5,3.5))
        ax2.barh([labels[i] for i in idx], imp[idx],
                 color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"])
        for i,v in enumerate(imp[idx]):
            ax2.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax2.set_xlabel("Önem Skoru"); plt.tight_layout(); st.pyplot(fig2)
 
    st.info("ℹ️ KNN ve SVM StandardScaler gerektirirken Karar Ağacı gerektirmez — bu fark Bölüm 3'te detaylı açıklanmıştır.")
 
# ════════════════════════════════════════════════════════════
# SEKME 2 — SINIFLANDIRMA / ARIZA TAHMİNİ
# ════════════════════════════════════════════════════════════
with t2:
    st.markdown("### 🔴 Makine Arızası Tahmini — AI4I 2020")
    st.divider()
 
    col_g, col_s = st.columns([1, 2])
 
    with col_g:
        st.markdown("#### ⚙️ Makine Sensör Değerleri")
        c_air  = st.slider("🌡️ Hava Sıcaklığı (K)", 295.0, 305.0, 300.0, step=0.1)
        c_proc = st.slider("🔥 İşlem Sıcaklığı (K)", 305.0, 315.0, 310.0, step=0.1)
        c_rpm  = st.slider("⚙️ Dönüş Hızı (rpm)", 1168, 2886, 1538)
        c_tork = st.slider("🔩 Tork (Nm)", 4.0, 77.0, 40.0, step=0.5)
        c_wear = st.slider("🔧 Takım Aşınması (dk)", 0, 250, 100)
        c_tip  = st.selectbox("🏷️ Ürün Tipi", ["L (Düşük)", "M (Orta)", "H (Yüksek)"])
        c_tip_e = {"L (Düşük)":0,"M (Orta)":1,"H (Yüksek)":2}[c_tip]
        st.divider()
        model_c = st.radio("🤖 Aktif Model", ["Decision Tree","KNN","SVM"], horizontal=True, key="mc")
 
    with col_s:
        @st.cache_data
        def predict_all_clf(air, proc, rpm, tork, wear, tip_e, _kaynak=None):
            yb = st.session_state.get("ai4i_bytes", None)
            df, _ = ai4i_verisi_yukle(yb)
            cols = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                                "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
            X = df[cols]; y = df["machine_failure"]
            Xtr,_,ytr,_ = train_test_split(X, y, test_size=0.2, random_state=42)
            sc = StandardScaler().fit(Xtr)
            vals = [tip_e, air, proc, rpm, tork, wear]
            girdi = pd.DataFrame([dict(zip(cols, vals[:len(cols)]))])
            girdi_s = sc.transform(girdi)
            dt  = DecisionTreeClassifier(max_depth=5,random_state=42).fit(Xtr,ytr)
            knn = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr),ytr)
            svc = SVC(kernel="rbf",C=1.0,probability=True,random_state=42).fit(sc.transform(Xtr),ytr)
            res = {}
            for name, m, g in [("dt",dt,girdi),("knn",knn,girdi_s),("svc",svc,girdi_s)]:
                pred = m.predict(g)[0]
                prob = m.predict_proba(g)[0][1]
                res[name] = {"pred":int(pred),"prob":float(prob)}
            return res
 
        with st.spinner("Hesaplanıyor..."):
            clf_res = predict_all_clf(c_air, c_proc, c_rpm, c_tork, c_wear, c_tip_e, _kaynak=sentetik_mi)
 
        aktif = clf_res[{"Decision Tree":"dt","KNN":"knn","SVM":"svc"}[model_c]]
 
        st.markdown(f"#### 🎯 Tahmin Sonucu — {model_c}")
        if aktif["pred"] == 1:
            st.error(f"🔴 **ARIZA RİSKİ YÜKSEKx** — Olasılık: %{aktif['prob']*100:.1f}")
        else:
            st.success(f"✅ **NORMAL ÇALIŞMA** — Arıza olasılığı: %{aktif['prob']*100:.1f}")
 
        st.progress(aktif["prob"], text=f"Arıza olasılığı: %{aktif['prob']*100:.1f}")
 
        d_dt  = "🔴 ARIZA" if clf_res["dt"]["pred"]==1  else "✅ Normal"
        d_knn = "🔴 ARIZA" if clf_res["knn"]["pred"]==1 else "✅ Normal"
        d_svc = "🔴 ARIZA" if clf_res["svc"]["pred"]==1 else "✅ Normal"
        st.info(f"🌳 DT: **{d_dt}** (%{clf_res['dt']['prob']*100:.1f}) &nbsp;|&nbsp; "
                f"🔵 KNN: **{d_knn}** (%{clf_res['knn']['prob']*100:.1f}) &nbsp;|&nbsp; "
                f"🟣 SVM: **{d_svc}** (%{clf_res['svc']['prob']*100:.1f})")
 
        st.divider()
        st.markdown("#### 📊 Model Performansı (Test Seti, n=2000)")
        cp = pd.DataFrame({
            "Model":    ["🌳 DT","🔵 KNN","🟣 SVM"],
            "Accuracy": [f"%{c_met['dt']['acc']}",f"%{c_met['knn']['acc']}",f"%{c_met['svc']['acc']}"],
            "Recall ⭐":[f"%{c_met['dt']['rec']}",f"%{c_met['knn']['rec']}",f"%{c_met['svc']['rec']}"],
            "Precision":[f"%{c_met['dt']['prec']}",f"%{c_met['knn']['prec']}",f"%{c_met['svc']['prec']}"],
            "F1":       [f"%{c_met['dt']['f1']}",f"%{c_met['knn']['f1']}",f"%{c_met['svc']['f1']}"],
        })
        st.dataframe(cp, hide_index=True, use_container_width=True)
        st.caption("⭐ Recall: Gerçek arızaların kaç %'i tespit edilebildi — üretimde en kritik metrik.")
 
    st.divider()
    st.markdown("#### 🗂️ Confusion Matrix (Test Seti)")
    cm1, cm2, cm3 = st.columns(3)
    for col, pred, name, cmap in [
        (cm1, dt_pred_c,  "Decision Tree", "Reds"),
        (cm2, knn_pred_c, "KNN",           "Blues"),
        (cm3, svc_pred_c, "SVM",           "Purples")
    ]:
        with col:
            cm = confusion_matrix(yte_c, pred)
            rec = round(recall_score(yte_c, pred, zero_division=0)*100,1)
            fig, ax = plt.subplots(figsize=(3.2,2.8))
            ax.imshow(cm, cmap=cmap)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Normal","Arıza"]); ax.set_yticklabels(["Normal","Arıza"])
            ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
            ax.set_title(f"{name}\nRecall: %{rec}", fontsize=10, fontweight="bold")
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,fontweight="bold",
                            color="white" if cm[i,j]>cm.max()/2 else "black")
            plt.tight_layout(); st.pyplot(fig)
 
# ════════════════════════════════════════════════════════════
# SEKME 3 — KARŞILAŞTIRMA
# ════════════════════════════════════════════════════════════
with t3:
    st.markdown("### 🆚 Üç Model Karşılaştırması")
    st.divider()
 
    st.markdown("#### 📈 Regresyon")
    st.dataframe(pd.DataFrame({
        "Metrik":          ["R²","MAE (TL)","RMSE (TL)","Ölçeklendirme","Özellik Önemi","Hız"],
        "🌳 Decision Tree":[r_met["dt"]["r2"],  f"{r_met['dt']['mae']:,.0f}",  f"{r_met['dt']['rmse']:,.0f}",  "❌ Gerekmez","✅ Var","⚡ Çok hızlı"],
        "🔵 KNN":          [r_met["knn"]["r2"], f"{r_met['knn']['mae']:,.0f}", f"{r_met['knn']['rmse']:,.0f}", "✅ Scaler","❌ Yok","🐢 Yavaş"],
        "🟣 SVM":          [r_met["svr"]["r2"], f"{r_met['svr']['mae']:,.0f}", f"{r_met['svr']['rmse']:,.0f}", "✅ Scaler","❌ Yok","⚡ Hızlı"],
    }), hide_index=True, use_container_width=True)
 
    best_r2 = max(r_met["dt"]["r2"], r_met["knn"]["r2"], r_met["svr"]["r2"])
    best_name = {r_met["dt"]["r2"]:"Decision Tree",r_met["knn"]["r2"]:"KNN",r_met["svr"]["r2"]:"SVM"}[best_r2]
    st.success(f"🏆 Regresyon R² kazananı: **{best_name}** (R²={best_r2})")
 
    st.divider()
    st.markdown("#### 🔴 Sınıflandırma")
    st.dataframe(pd.DataFrame({
        "Metrik":          ["Accuracy","Recall ⭐","Precision","F1"],
        "🌳 Decision Tree":[f"%{c_met['dt']['acc']}",f"%{c_met['dt']['rec']}",f"%{c_met['dt']['prec']}",f"%{c_met['dt']['f1']}"],
        "🔵 KNN":          [f"%{c_met['knn']['acc']}",f"%{c_met['knn']['rec']}",f"%{c_met['knn']['prec']}",f"%{c_met['knn']['f1']}"],
        "🟣 SVM":          [f"%{c_met['svc']['acc']}",f"%{c_met['svc']['rec']}",f"%{c_met['svc']['prec']}",f"%{c_met['svc']['f1']}"],
    }), hide_index=True, use_container_width=True)
 
    best_rec = max(c_met["dt"]["rec"], c_met["knn"]["rec"], c_met["svc"]["rec"])
    best_rname = {c_met["dt"]["rec"]:"Decision Tree",c_met["knn"]["rec"]:"KNN",c_met["svc"]["rec"]:"SVM"}[best_rec]
    st.success(f"🏆 Recall kazananı: **{best_rname}** (%{best_rec})")
 
    st.divider()
    st.markdown("#### 🔬 Teorik Karşılaştırma")
    st.dataframe(pd.DataFrame({
        "Özellik":    ["Paradigma","Eğitim","Tahmin Hızı","Bellek","Ölçeklendirme","Yorumlanabilirlik","Hiperparametre"],
        "🌳 DT":      ["Eager/Kural","Ağaç öğrenir","O(log n)","Düşük","❌","⭐⭐⭐","max_depth"],
        "🔵 KNN":     ["Lazy/Örnek","Veriyi saklar","O(n·d)","Yüksek","✅ Zorunlu","⭐","K"],
        "🟣 SVM":     ["Eager/Marjin","SV bulur","O(sv·d)","Orta","✅ Zorunlu","⭐","C, kernel"],
    }), hide_index=True, use_container_width=True)
 
# ════════════════════════════════════════════════════════════
# SEKME 4 — HİPERPARAMETRE ANALİZİ
# ════════════════════════════════════════════════════════════
with t4:
    st.markdown("### 🔍 Hiperparametre Optimizasyon Analizi")
    st.divider()
 
    hp_sec = st.radio("Analiz:", ["KNN — K Optimizasyonu","SVM — C Optimizasyonu"], horizontal=True)
 
    @st.cache_data
    def k_opt_reg():
        df = sentetik_regression_verisi()
        X = df[["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]]
        y = df["toplam_maliyet"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr)
        Xtr_s,Xte_s = sc.transform(Xtr),sc.transform(Xte)
        tr,te = [],[]
        for k in range(1,21):
            m = KNeighborsRegressor(n_neighbors=k).fit(Xtr_s,ytr)
            tr.append(r2_score(ytr,m.predict(Xtr_s)))
            te.append(r2_score(yte,m.predict(Xte_s)))
        return tr,te
 
    @st.cache_data
    def k_opt_clf(_kaynak=None):
        yb = st.session_state.get("ai4i_bytes", None)
        df,_ = ai4i_verisi_yukle(yb)
        cols = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                            "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
        X = df[cols]; y = df["machine_failure"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr)
        Xtr_s,Xte_s = sc.transform(Xtr),sc.transform(Xte)
        tr,te = [],[]
        for k in range(1,21):
            m = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s,ytr)
            tr.append(accuracy_score(ytr,m.predict(Xtr_s)))
            te.append(accuracy_score(yte,m.predict(Xte_s)))
        return tr,te
 
    @st.cache_data
    def c_opt_reg():
        df = sentetik_regression_verisi()
        X = df[["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]]
        y = df["toplam_maliyet"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr)
        Xtr_s,Xte_s = sc.transform(Xtr),sc.transform(Xte)
        c_vals = [0.01,0.1,1,10,50,100,200,500,1000]
        tr,te = [],[]
        for c in c_vals:
            m = SVR(kernel="rbf",C=c).fit(Xtr_s,ytr)
            tr.append(r2_score(ytr,m.predict(Xtr_s)))
            te.append(r2_score(yte,m.predict(Xte_s)))
        return c_vals,tr,te
 
    @st.cache_data
    def c_opt_clf(_kaynak=None):
        yb = st.session_state.get("ai4i_bytes", None)
        df,_ = ai4i_verisi_yukle(yb)
        cols = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                            "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
        X = df[cols]; y = df["machine_failure"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr)
        Xtr_s,Xte_s = sc.transform(Xtr),sc.transform(Xte)
        c_vals = [0.01,0.1,1,10,50,100,200,500,1000]
        tr,te = [],[]
        for c in c_vals:
            m = SVC(kernel="rbf",C=c,random_state=42).fit(Xtr_s,ytr)
            tr.append(accuracy_score(ytr,m.predict(Xtr_s)))
            te.append(accuracy_score(yte,m.predict(Xte_s)))
        return c_vals,tr,te
 
    if "KNN" in hp_sec:
        with st.spinner("K analizi hesaplanıyor..."):
            tr_r,te_r = k_opt_reg()
            tr_c,te_c = k_opt_clf(_kaynak=sentetik_mi)
        h1c, h2c = st.columns(2)
        with h1c:
            best_k = int(np.argmax(te_r))+1
            st.markdown(f"#### Regresyon — K vs R²  (En iyi K={best_k})")
            fig,ax = plt.subplots(figsize=(6,4))
            ax.plot(range(1,21),tr_r,"o-",color="#e74c3c",label="Eğitim R²",lw=2)
            ax.plot(range(1,21),te_r,"s-",color="#3498db",label="Test R²",lw=2)
            ax.axvline(best_k,color="green",linestyle="--",lw=1.5,label=f"En iyi K={best_k}")
            ax.axvline(5,color="gray",linestyle=":",lw=1,label="Kullanılan K=5")
            ax.set_xlabel("K"); ax.set_ylabel("R²"); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title("K=1 → Overfitting  |  Büyük K → Underfitting",fontsize=10)
            plt.tight_layout(); st.pyplot(fig)
        with h2c:
            best_kc = int(np.argmax(te_c))+1
            st.markdown(f"#### Sınıflandırma — K vs Accuracy  (En iyi K={best_kc})")
            fig,ax = plt.subplots(figsize=(6,4))
            ax.plot(range(1,21),tr_c,"o-",color="#e74c3c",label="Eğitim Acc",lw=2)
            ax.plot(range(1,21),te_c,"s-",color="#3498db",label="Test Acc",lw=2)
            ax.axvline(best_kc,color="green",linestyle="--",lw=1.5,label=f"En iyi K={best_kc}")
            ax.axvline(5,color="gray",linestyle=":",lw=1,label="Kullanılan K=5")
            ax.set_xlabel("K"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title("K=1 → Overfitting  |  Büyük K → Underfitting",fontsize=10)
            plt.tight_layout(); st.pyplot(fig)
    else:
        with st.spinner("C analizi hesaplanıyor..."):
            c_vals,tr_r,te_r = c_opt_reg()
            c_vals,tr_c,te_c = c_opt_clf(_kaynak=sentetik_mi)
        h1c, h2c = st.columns(2)
        c_labels = [str(c) for c in c_vals]
        with h1c:
            best_c = c_vals[int(np.argmax(te_r))]
            st.markdown(f"#### Regresyon (SVR) — C vs R²  (En iyi C={best_c})")
            fig,ax = plt.subplots(figsize=(6,4))
            ax.plot(range(len(c_vals)),tr_r,"o-",color="#e74c3c",label="Eğitim R²",lw=2)
            ax.plot(range(len(c_vals)),te_r,"s-",color="#9b59b6",label="Test R²",lw=2)
            ax.set_xticks(range(len(c_vals))); ax.set_xticklabels(c_labels,rotation=45,fontsize=8)
            ax.set_xlabel("C"); ax.set_ylabel("R²"); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title("C küçük → Underfitting  |  C büyük → Overfitting",fontsize=10)
            plt.tight_layout(); st.pyplot(fig)
        with h2c:
            best_cc = c_vals[int(np.argmax(te_c))]
            st.markdown(f"#### Sınıflandırma (SVC) — C vs Acc  (En iyi C={best_cc})")
            fig,ax = plt.subplots(figsize=(6,4))
            ax.plot(range(len(c_vals)),tr_c,"o-",color="#e74c3c",label="Eğitim Acc",lw=2)
            ax.plot(range(len(c_vals)),te_c,"s-",color="#9b59b6",label="Test Acc",lw=2)
            ax.set_xticks(range(len(c_vals))); ax.set_xticklabels(c_labels,rotation=45,fontsize=8)
            ax.set_xlabel("C"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title("C küçük → Underfitting  |  C büyük → Overfitting",fontsize=10)
            plt.tight_layout(); st.pyplot(fig)
 
    st.divider()
    st.dataframe(pd.DataFrame({
        "Durum":   ["Overfitting","Underfitting","Optimal"],
        "🌳 DT":   ["max_depth büyük","max_depth küçük","Test kaybı minimum"],
        "🔵 KNN":  ["K=1","K çok büyük","Test R²/Acc maksimum"],
        "🟣 SVM":  ["C çok büyük","C çok küçük","Test performansı maksimum"],
    }), hide_index=True, use_container_width=True)
 
# ════════════════════════════════════════════════════════════
# SEKME 5 — ALGORİTMA TEORİSİ
# ════════════════════════════════════════════════════════════
with t5:
    st.markdown("### 📚 Algoritma Teorisi")
    st.divider()
 
    a1,a2,a3 = st.tabs(["🌳 Decision Tree","🔵 KNN","🟣 SVM"])
 
    with a1:
        st.markdown("""
**Paradigma:** Eager Learning — Kural Tabanlı
 
Veri uzayını özyinelemeli ikili bölümlere ayırarak kural üretir.
- **Regresyon:** Her yaprakta MSE minimize → yaprak ortalaması döndürülür
- **Sınıflandırma:** Gini safsızlık indeksi minimize → çoğunluk sınıfı
- **Ölçeklendirme gerekmez:** Kararlar `x < eşik` sıralama ilişkisine dayanır
- **Özellik önemi:** `feature_importances_` ile hesaplanabilir — yorumlanabilirlik yüksek
 
**max_depth küçük** → Underfitting | **max_depth büyük** → Overfitting
        """)
 
    with a2:
        st.markdown("""
**Paradigma:** Lazy Learning — Örnek Tabanlı
 
Model eğitmez, tüm veriyi bellekte tutar. Tahmin anında mesafe hesaplar.
- **Regresyon:** K komşunun ortalaması
- **Sınıflandırma:** K komşunun çoğunluk oyu
- **Mesafe:** Öklid — `d = √Σ(xᵢ−xⱼ)²`
 
**Ölçeklendirme ZORUNLU:** rpm≈1538 ile hammadde≈45 TL aynı formüle girerse rpm mesafeyi 34× baskılar.  
StandardScaler: `z = (x−μ)/σ` → her değişken μ=0, σ=1
 
**K=1** → Overfitting | **K büyük** → Underfitting
        """)
 
    with a3:
        st.markdown("""
**Paradigma:** Eager Learning — Marjin Tabanlı (Geometrik)
 
Sınıflar arasındaki **maksimum marjini** bulan hiper düzlem öğrenir.  
Marjini belirleyen noktalara **Support Vectors** denir.
 
- **C parametresi:** Küçük C = geniş marjin + hata toleransı | Büyük C = dar marjin + sıfır hata isteği
- **RBF Kernel:** `K(x,y) = exp(−γ||x−y||²)` — doğrusal olmayan sınırları modeleller
- **SVR (Regresyon):** ε-duyarsız tüp — tüp içindeki hatalar cezalandırılmaz
 
**Ölçeklendirme ZORUNLU:** Marjin hesabı mesafeye dayalıdır — KNN ile aynı gerekçe.
 
**C küçük** → Underfitting | **C büyük** → Overfitting
        """)
 
    st.divider()
    st.markdown("#### Hangi durumda hangi algoritma?")
    st.dataframe(pd.DataFrame({
        "Kriter":  ["Yorumlanabilirlik şart","Gerçek zamanlı tahmin","Büyük veri (>100K)","Az veri","Doğrusal olmayan ilişki"],
        "Öneri":   ["🌳 Decision Tree","🌳 DT veya 🟣 SVM","🌳 DT veya 🟣 SVM","🟣 SVM","🟣 SVM (kernel) veya 🌳 DT"],
    }), hide_index=True, use_container_width=True)
