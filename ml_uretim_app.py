# ============================================================
#   ÜRETİM ZEKÂ PLATFORMU
#   Decision Tree Regression + Classification
#
#   KURULUM:
#       python -m pip install streamlit scikit-learn pandas numpy matplotlib seaborn
#
#   ÇALIŞTIRMA:
#       streamlit run ml_uretim_app.py
#
#   VERİ SETİ (opsiyonel):
#       AI4I 2020 → ai4i2020.csv dosyasını aynı klasöre koyun
#       Yoksa uygulama sentetik veriyle çalışır
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import os, warnings
warnings.filterwarnings("ignore")

# ── Sayfa Ayarları ──────────────────────────────────────────
st.set_page_config(
    page_title="Üretim Zekâ Platformu",
    page_icon="🏭",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 800; }
    .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 600; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ── VERİ SETLERİ ────────────────────────────────────────────
# ============================================================

@st.cache_data
def sentetik_regression_verisi():
    """Üretim maliyeti tahmini için sentetik veri."""
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
    """AI4I 2020 verisini yükler, yoksa sentetik üretir."""
    if yuklenen is not None:
        df = pd.read_csv(yuklened)
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
        return df, False   # False = gerçek veri

    # Sentetik AI4I verisi
    np.random.seed(42)
    n = 10000
    tip    = np.random.choice(["L","M","H"], n, p=[0.5,0.3,0.2])
    tip_e  = np.where(tip=="L",0,np.where(tip=="M",1,2))
    air    = np.random.normal(300, 2, n)
    proc   = air + np.random.normal(10, 1, n)
    rpm    = np.random.normal(1538, 179, n).clip(1168, 2886)
    tork   = np.random.normal(40, 10, n).clip(3.8, 76.6)
    wear   = (tip_e*30 + np.random.uniform(0,200,n)).clip(0,250)
    power  = tork * rpm * 2 * np.pi / 60
    failure = ((power > 9000) | (tork*(250-wear)>13000) |
               (rpm<1380)&(tork>40) | (np.random.rand(n)<0.001)).astype(int)
    df = pd.DataFrame({
        "type": tip, "type_encoded": tip_e,
        "air_temperature_k": air.round(1),
        "process_temperature_k": proc.round(1),
        "rotational_speed_rpm": rpm.round(0).astype(int),
        "torque_nm": tork.round(1),
        "tool_wear_min": wear.round(0).astype(int),
        "machine_failure": failure,
    })
    return df, True   # True = sentetik


@st.cache_resource
def regression_modeli_egit(df):
    X = df[["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]]
    y = df["toplam_maliyet"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    return model, Xtr, Xte, ytr, yte, ypred, {
        "r2": round(r2_score(yte, ypred), 4),
        "mae": round(mean_absolute_error(yte, ypred), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, ypred))), 0),
    }


@st.cache_resource
def classification_modeli_egit(df):
    ozellikler = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                               "rotational_speed_rpm","torque_nm","tool_wear_min"]
                  if c in df.columns]
    X = df[ozellikler]
    y = df["machine_failure"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    return model, Xtr, Xte, ytr, yte, ypred, {
        "accuracy": round(accuracy_score(yte, ypred)*100, 2),
        "ozellikler": ozellikler,
    }


# ============================================================
# ── VERİ YÜKLE ──────────────────────────────────────────────
# ============================================================
reg_df  = sentetik_regression_verisi()
ai4i_df, sentetik_mi = ai4i_verisi_yukle()

reg_model,  Xtr_r, Xte_r, ytr_r, yte_r, ypred_r, reg_met  = regression_modeli_egit(reg_df)
clf_model,  Xtr_c, Xte_c, ytr_c, yte_c, ypred_c, clf_met  = classification_modeli_egit(ai4i_df)


# ============================================================
# ── BAŞLIK ──────────────────────────────────────────────────
# ============================================================
col_t1, col_t2 = st.columns([3,1])
with col_t1:
    st.title("🏭 Üretim Zekâ Platformu")
    st.markdown("**Decision Tree** tabanlı makine öğrenmesi — Regression & Classification")
with col_t2:
    if sentetik_mi:
        st.warning("⚠️ Demo mod\nai4i2020.csv bulunamadı")
    else:
        st.success("✅ AI4I 2020\nGerçek veri yüklendi")

st.divider()

# ── AI4I CSV yükleme alanı (küçük, collapsible) ─────────────
with st.expander("📂 AI4I 2020 CSV Yükle (opsiyonel)"):
    yuklenen = st.file_uploader("ai4i2020.csv", type=["csv"])
    if yuklenen:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# ── ANA SEKMELER ────────────────────────────────────────────
# ============================================================
ana1, ana2, ana3 = st.tabs([
    "📈 REGRESSION — Maliyet Tahmini",
    "🔴 CLASSIFICATION — Arıza Tahmini",
    "📊 Model Karşılaştırma"
])


# ════════════════════════════════════════════════════════════
# ANA SEKME 1: REGRESSION
# ════════════════════════════════════════════════════════════
with ana1:
    st.markdown("### 📈 Üretim Maliyeti Tahmini")
    st.markdown("*Sentetik üretim verisi (n=500) — Decision Tree Regressor (max_depth=5)*")
    st.divider()

    # Sidebar benzeri giriş alanı
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
        # Model metrikleri her zaman göster
        st.markdown("#### 📊 Model Performansı")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Skoru", reg_met["r2"], help="1'e yakın = iyi")
        m2.metric("MAE", f"{reg_met['mae']:,.0f} TL")
        m3.metric("RMSE", f"{reg_met['rmse']:,.0f} TL")
        st.divider()

        if r_btn:
            girdi = pd.DataFrame([{
                "uretim_miktari": r_uretim, "makine_yasi": r_makine,
                "iscilik_saati": r_iscilik, "hammadde_fiyati": r_hammadde,
                "vardiya": r_vval
            }])
            tahmin = reg_model.predict(girdi)[0]
            birim  = tahmin / r_uretim

            c1, c2 = st.columns(2)
            c1.metric("💰 Tahmini Maliyet", f"{tahmin:,.0f} TL")
            c2.metric("📌 Birim Maliyet",   f"{birim:,.0f} TL/adet")

            if tahmin > 25000:
                st.error(f"⚠️ Yüksek maliyet — {tahmin:,.0f} TL. Optimizasyon önerilir.")
            elif tahmin > 15000:
                st.warning(f"🟡 Orta maliyet — {tahmin:,.0f} TL.")
            else:
                st.success(f"✅ Normal maliyet — {tahmin:,.0f} TL.")
        else:
            st.info("← Parametreleri ayarlayıp **Maliyet Tahmin Et** butonuna tıklayın.")

    st.divider()

    # Grafikler
    gc1, gc2 = st.columns(2)

    with gc1:
        st.markdown("##### 🎯 Gerçek vs Tahmin")
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.scatter(yte_r, ypred_r, alpha=0.4, s=15, color="#e74c3c")
        mn, mx = float(yte_r.min()), float(yte_r.max())
        ax.plot([mn,mx],[mn,mx],"k--",lw=1.5, label="Mükemmel")
        ax.set_xlabel("Gerçek (TL)"); ax.set_ylabel("Tahmin (TL)")
        ax.set_title(f"R² = {reg_met['r2']}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig)

    with gc2:
        st.markdown("##### 📌 Özellik Önemi")
        imp = reg_model.feature_importances_
        fnames = ["Üretim\nMiktarı","Makine\nYaşı","İşçilik\nSaati","Hammadde\nFiyatı","Vardiya"]
        idx = np.argsort(imp)
        fig2, ax2 = plt.subplots(figsize=(5,3.5))
        colors = ["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"]
        ax2.barh([fnames[i] for i in idx], imp[idx],
                  color=[colors[i] for i in idx], edgecolor="white")
        for i, v in enumerate(imp[idx]):
            ax2.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax2.set_xlabel("Önem Skoru")
        ax2.set_title("Hangi özellik daha etkili?", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig2)

    # Overfitting
    st.markdown("##### 🔍 Overfitting Analizi")
    tr2_list, te2_list = [], []
    for d in range(1,16):
        m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(Xtr_r, ytr_r)
        tr2_list.append(r2_score(ytr_r, m.predict(Xtr_r)))
        te2_list.append(r2_score(yte_r, m.predict(Xte_r)))
    chart_df = pd.DataFrame({"Eğitim R²": tr2_list, "Test R²": te2_list}, index=range(1,16))
    chart_df.index.name = "max_depth"
    st.line_chart(chart_df, height=220)

    # Karar ağacı kuralları
    with st.expander("🌳 Karar Ağacı Kuralları (depth=3)"):
        sm = DecisionTreeRegressor(max_depth=3, random_state=42).fit(Xtr_r, ytr_r)
        st.code(export_text(sm, feature_names=list(Xtr_r.columns)), language="text")


# ════════════════════════════════════════════════════════════
# ANA SEKME 2: CLASSIFICATION
# ════════════════════════════════════════════════════════════
with ana2:
    st.markdown("### 🔴 Makine Arıza Tahmini (Classification)")
    veri_tipi = "Sentetik" if sentetik_mi else "AI4I 2020 Gerçek Veri"
    st.markdown(f"*{veri_tipi} (n={len(ai4i_df):,}) — Decision Tree Classifier (max_depth=5)*")
    st.divider()

    col_gir2, col_sonu2 = st.columns([1, 2])

    with col_gir2:
        st.markdown("#### ⚙️ Parametreler")
        c_tip     = st.selectbox("🏷️ Ürün Tipi", ["L — Düşük","M — Orta","H — Yüksek"], key="c_tip")
        c_tipval  = {"L":0,"M":1,"H":2}[c_tip[0]]
        c_air     = st.slider("🌡️ Hava Sıcaklığı (K)", 295.0, 305.0, 300.0, step=0.1, key="c_air")
        c_proc    = st.slider("🔥 Proses Sıcaklığı (K)", 305.0, 315.0, 310.0, step=0.1, key="c_proc")
        c_rpm     = st.slider("⚡ Dönüş Hızı (rpm)", 1168, 2886, 1538, key="c_rpm")
        c_tork    = st.slider("🔩 Tork (Nm)", 3.8, 76.6, 40.0, step=0.1, key="c_tork")
        c_wear    = st.slider("🔧 Takım Aşınması (dk)", 0, 250, 100, key="c_wear")
        st.divider()
        c_btn = st.button("🔍 Arıza Riskini Tahmin Et", type="primary", use_container_width=True, key="c_btn")

    with col_sonu2:
        st.markdown("#### 📊 Model Performansı")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Doğruluk (Accuracy)", f"%{clf_met['accuracy']}")
        cm2.metric("Eğitim Seti", f"{len(Xtr_c):,} kayıt")
        cm3.metric("Test Seti",   f"{len(Xte_c):,} kayıt")
        st.divider()

        if c_btn:
            girdi2_dict = {}
            for col in clf_met["ozellikler"]:
                if "type" in col:   girdi2_dict[col] = c_tipval
                elif "air" in col:  girdi2_dict[col] = c_air
                elif "process" in col: girdi2_dict[col] = c_proc
                elif "speed" in col or "rpm" in col: girdi2_dict[col] = c_rpm
                elif "torque" in col or "tork" in col: girdi2_dict[col] = c_tork
                elif "wear" in col: girdi2_dict[col] = c_wear

            girdi2 = pd.DataFrame([girdi2_dict])
            pred   = clf_model.predict(girdi2)[0]
            prob   = clf_model.predict_proba(girdi2)[0]

            if pred == 1:
                st.error(f"🔴 **ARIZ A RİSKİ YÜKSEK**\nArıza olasılığı: %{prob[1]*100:.1f}")
            else:
                st.success(f"✅ **NORMAL ÇALIŞMA**\nArıza olasılığı: %{prob[1]*100:.1f}")

            st.progress(float(prob[1]), text=f"Arıza olasılığı: %{prob[1]*100:.1f}")
        else:
            st.info("← Makine parametrelerini girin ve **Arıza Riskini Tahmin Et** butonuna tıklayın.")

    st.divider()

    gc3, gc4 = st.columns(2)

    with gc3:
        st.markdown("##### 📌 Özellik Önemi")
        imp2   = clf_model.feature_importances_
        flabs2 = [c.replace("_"," ").title() for c in clf_met["ozellikler"]]
        idx2   = np.argsort(imp2)
        fig3, ax3 = plt.subplots(figsize=(5,3.5))
        ax3.barh([flabs2[i] for i in idx2], imp2[idx2],
                  color="#e74c3c", edgecolor="white", alpha=0.85)
        for i, v in enumerate(imp2[idx2]):
            ax3.text(v+0.003, i, f"{v:.3f}", va="center", fontsize=9)
        ax3.set_xlabel("Önem Skoru")
        ax3.set_title("Arızayı etkileyen faktörler", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig3)

    with gc4:
        st.markdown("##### 🗂️ Confusion Matrix")
        cm = confusion_matrix(yte_c, ypred_c)
        fig4, ax4 = plt.subplots(figsize=(4,3.5))
        im = ax4.imshow(cm, cmap="Reds")
        ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
        ax4.set_xticklabels(["Normal","Arıza"])
        ax4.set_yticklabels(["Normal","Arıza"])
        ax4.set_xlabel("Tahmin"); ax4.set_ylabel("Gerçek")
        ax4.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, str(cm[i,j]), ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if cm[i,j] > cm.max()/2 else "black")
        plt.tight_layout(); st.pyplot(fig4)

    # Classification report
    with st.expander("📋 Detaylı Classification Report"):
        report = classification_report(yte_c, ypred_c,
                                        target_names=["Normal","Arıza"])
        st.code(report)

    # Karar ağacı
    with st.expander("🌳 Karar Ağacı Görselleştirmesi (depth=3)"):
        sm2 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr_c, ytr_c)
        fig5, ax5 = plt.subplots(figsize=(16,6))
        plot_tree(sm2, feature_names=clf_met["ozellikler"],
                  class_names=["Normal","Arıza"],
                  filled=True, rounded=True, fontsize=8,
                  ax=ax5, impurity=False)
        ax5.set_title("Classification Tree (depth=3)", fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig5)


# ════════════════════════════════════════════════════════════
# ANA SEKME 3: MODEL KARŞILAŞTIRMA
# ════════════════════════════════════════════════════════════
with ana3:
    st.markdown("### 📊 Model Karşılaştırma Özeti")
    st.divider()

    col_r, col_c = st.columns(2)

    with col_r:
        st.markdown("#### 📈 Regression Modeli")
        st.markdown("**Görev:** Üretim maliyeti tahmini (TL)")
        st.markdown("**Veri:** Sentetik üretim verisi (n=500)")
        st.markdown("**Algoritma:** Decision Tree Regressor")

        rd = pd.DataFrame({
            "Metrik": ["R² Skoru","MAE","RMSE","max_depth","Eğitim seti","Test seti"],
            "Değer":  [reg_met["r2"],
                       f"{reg_met['mae']:,.0f} TL",
                       f"{reg_met['rmse']:,.0f} TL",
                       "5",
                       f"{len(Xtr_r)} kayıt",
                       f"{len(Xte_r)} kayıt"]
        })
        st.dataframe(rd, hide_index=True, use_container_width=True)

        st.markdown("**En önemli özellik:**")
        top_r = list(Xtr_r.columns)[np.argmax(reg_model.feature_importances_)]
        st.info(f"🏆 `{top_r}` — önem: {max(reg_model.feature_importances_):.3f}")

    with col_c:
        st.markdown("#### 🔴 Classification Modeli")
        st.markdown(f"**Görev:** Makine arıza tahmini (0/1)")
        st.markdown(f"**Veri:** {'Sentetik' if sentetik_mi else 'AI4I 2020'} (n={len(ai4i_df):,})")
        st.markdown("**Algoritma:** Decision Tree Classifier")

        cd = pd.DataFrame({
            "Metrik": ["Accuracy","max_depth","Eğitim seti","Test seti",
                       "Arıza oranı (test)","Normal oranı (test)"],
            "Değer":  [f"%{clf_met['accuracy']}",
                       "5",
                       f"{len(Xtr_c):,} kayıt",
                       f"{len(Xte_c):,} kayıt",
                       f"%{(yte_c==1).mean()*100:.1f}",
                       f"%{(yte_c==0).mean()*100:.1f}"]
        })
        st.dataframe(cd, hide_index=True, use_container_width=True)

        st.markdown("**En önemli özellik:**")
        top_c = clf_met["ozellikler"][np.argmax(clf_model.feature_importances_)]
        st.info(f"🏆 `{top_c}` — önem: {max(clf_model.feature_importances_):.3f}")

    st.divider()
    st.markdown("#### 🔍 Her İki Model İçin Overfitting Analizi")
    oa1, oa2 = st.columns(2)

    with oa1:
        st.caption("Regression — R² vs max_depth")
        tr2, te2 = [], []
        for d in range(1,16):
            m = DecisionTreeRegressor(max_depth=d,random_state=42).fit(Xtr_r,ytr_r)
            tr2.append(r2_score(ytr_r,m.predict(Xtr_r)))
            te2.append(r2_score(yte_r,m.predict(Xte_r)))
        st.line_chart(pd.DataFrame({"Eğitim R²":tr2,"Test R²":te2},
                                    index=range(1,16)), height=200)

    with oa2:
        st.caption("Classification — Accuracy vs max_depth")
        tra, tea = [], []
        for d in range(1,16):
            m = DecisionTreeClassifier(max_depth=d,random_state=42).fit(Xtr_c,ytr_c)
            tra.append(accuracy_score(ytr_c,m.predict(Xtr_c)))
            tea.append(accuracy_score(yte_c,m.predict(Xte_c)))
        st.line_chart(pd.DataFrame({"Eğitim Acc":tra,"Test Acc":tea},
                                    index=range(1,16)), height=200)

    st.divider()
    st.caption("📚 Veri Kaynakları: Sentetik üretim verisi (Regression) | "
               "AI4I 2020 Predictive Maintenance Dataset — Matzka (2020) (Classification)")
