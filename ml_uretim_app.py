import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, io
warnings.filterwarnings("ignore")
 
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
 
st.set_page_config(page_title="Üretim Zekâ Platformu", page_icon="🏭", layout="wide")
 
# ─────────────────────────────────────────────
# VERİ FONKSİYONLARI
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def uretim_verisi():
    np.random.seed(42)
    n = 500
    ur  = np.random.randint(50, 500, n)
    mk  = np.random.randint(1, 20, n)
    is_ = np.round(np.random.uniform(4, 16, n), 1)
    hm  = np.round(np.random.uniform(10, 80, n), 1)
    vr  = np.random.choice([1, 2], n)
    mal = ur*hm*0.9 + is_*250 + mk*150 + (vr==2)*1200 + np.random.normal(0,500,n)
    return pd.DataFrame({"uretim_miktari":ur,"makine_yasi":mk,"iscilik_saati":is_,
                          "hammadde_fiyati":hm,"vardiya":vr,"toplam_maliyet":np.round(mal,0)})
 
@st.cache_data(show_spinner=False)
def ai4i_indir():
    try:
        import requests
        r = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
            timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = (df.columns.str.strip()
                          .str.replace(r"[\[\]()]","",regex=True)
                          .str.replace(" ","_").str.lower())
            if "type" in df.columns:
                df["type_encoded"] = LabelEncoder().fit_transform(df["type"])
            return df
    except Exception:
        pass
    return None
 
def ai4i_sentetik():
    np.random.seed(42)
    n = 10000
    tip  = np.random.choice(["L","M","H"], n, p=[0.5,0.3,0.2])
    te   = np.where(tip=="L",0,np.where(tip=="M",1,2))
    air  = np.random.normal(300,2,n)
    proc = air + np.random.normal(10,1,n)
    rpm  = np.random.normal(1538,179,n).clip(1168,2886)
    tork = np.random.normal(40,10,n).clip(3.8,76.6)
    wear = (te*30 + np.random.uniform(0,200,n)).clip(0,250)
    power = tork*rpm*2*np.pi/60
    fail  = ((power>9000)|(tork*(250-wear)>13000)|
             ((rpm<1380)&(tork>40))|(np.random.rand(n)<0.001)).astype(int)
    return pd.DataFrame({"type":tip,"type_encoded":te,
        "air_temperature_k":air.round(1),"process_temperature_k":proc.round(1),
        "rotational_speed_rpm":rpm.round(0).astype(int),
        "torque_nm":tork.round(1),"tool_wear_min":wear.round(0).astype(int),
        "machine_failure":fail})
 
# ─────────────────────────────────────────────
# MODEL FONKSİYONLARI
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def reg_egit():
    df   = uretim_verisi()
    feat = ["uretim_miktari","makine_yasi","iscilik_saati","hammadde_fiyati","vardiya"]
    X, y = df[feat], df["toplam_maliyet"]
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
 
    dt  = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr, ytr)
    knn = KNeighborsRegressor(n_neighbors=5).fit(Xtr_s, ytr)
    svm = SVR(kernel="rbf", C=100, epsilon=0.1).fit(Xtr_s, ytr)
 
    def met(m, g):
        p = m.predict(g)
        return {"r2":round(r2_score(yte,p),4),
                "mae":round(mean_absolute_error(yte,p),0),
                "rmse":round(float(np.sqrt(mean_squared_error(yte,p))),0),
                "pred":p.tolist()}
 
    return ({"dt":dt,"knn":knn,"svm":svm},
            sc, feat,
            Xtr, Xte, ytr, yte.tolist(),
            {"dt":met(dt,Xte),"knn":met(knn,Xte_s),"svm":met(svm,Xte_s)},
            dt.feature_importances_.tolist())
 
@st.cache_data(show_spinner=False)
def clf_egit(veri_hash):
    df = st.session_state.get("ai4i_df")
    if df is None:
        df = ai4i_sentetik()
    feat = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                         "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
    X, y = df[feat], df["machine_failure"]
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
 
    dt  = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr, ytr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ytr)
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42).fit(Xtr_s, ytr)
 
    def met(m, g):
        p = m.predict(g)
        return {"acc":round(accuracy_score(yte,p)*100,2),
                "rec":round(recall_score(yte,p,zero_division=0)*100,1),
                "prec":round(precision_score(yte,p,zero_division=0)*100,1),
                "f1":round(f1_score(yte,p,zero_division=0)*100,1),
                "pred":p.tolist()}
 
    return ({"dt":dt,"knn":knn,"svm":svm},
            sc, feat,
            Xtr, Xte, ytr, yte.tolist(),
            {"dt":met(dt,Xte),"knn":met(knn,Xte_s),"svm":met(svm,Xte_s)})
 
# ─────────────────────────────────────────────
# AI4I VERİSİ YÜKLE (session state)
# ─────────────────────────────────────────────
if "ai4i_df" not in st.session_state:
    with st.spinner("AI4I 2020 verisi UCI'dan indiriliyor..."):
        df_ind = ai4i_indir()
    if df_ind is not None:
        st.session_state["ai4i_df"]     = df_ind
        st.session_state["ai4i_kaynak"] = "uci"
    else:
        st.session_state["ai4i_df"]     = None
        st.session_state["ai4i_kaynak"] = "sentetik"
 
kaynak = st.session_state.get("ai4i_kaynak", "sentetik")
 
# ─────────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────────
st.title("🏭 Üretim Zekâ Platformu")
st.markdown("**Decision Tree · KNN · SVM** — Üretim Maliyeti Tahmini & Makine Arızası Tespiti")
 
if kaynak == "uci":
    st.success("✅ AI4I 2020 gerçek verisi yüklendi (UCI) — 10.000 gözlem")
elif kaynak == "yuklendi":
    st.success("✅ AI4I 2020 verisi yüklendi (CSV)")
else:
    st.warning("⚠️ AI4I verisi indirilemedi — demo mod (sentetik veri)")
    with st.expander("📂 ai4i2020.csv yükle"):
        st.markdown("Veriyi [UCI](https://archive.ics.uci.edu/dataset/601) sayfasından indirip buraya yükleyebilirsiniz.")
        yuklenen = st.file_uploader("ai4i2020.csv", type=["csv"])
        if yuklenen:
            raw = yuklenen.read()
            df_up = pd.read_csv(io.BytesIO(raw))
            df_up.columns = (df_up.columns.str.strip()
                             .str.replace(r"[\[\]()]","",regex=True)
                             .str.replace(" ","_").str.lower())
            if "type" in df_up.columns:
                df_up["type_encoded"] = LabelEncoder().fit_transform(df_up["type"])
            st.session_state["ai4i_df"]     = df_up
            st.session_state["ai4i_kaynak"] = "yuklendi"
            clf_egit.clear()
            st.rerun()
 
st.divider()
 
# ─────────────────────────────────────────────
# MODELLERİ EĞİT
# ─────────────────────────────────────────────
with st.spinner("Modeller hazırlanıyor..."):
    (r_mdl, r_sc, r_feat, Xtr_r, Xte_r, ytr_r, yte_r_lst,
     r_met, r_imp) = reg_egit()
 
    veri_hash = kaynak + str(len(st.session_state.get("ai4i_df") or []))
    (c_mdl, c_sc, c_feat, Xtr_c, Xte_c, ytr_c, yte_c_lst,
     c_met) = clf_egit(veri_hash)
 
yte_r = np.array(yte_r_lst)
yte_c = np.array(yte_c_lst)
 
# ─────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📈 Maliyet Tahmini",
    "🔴 Arıza Tahmini",
    "🆚 Model Karşılaştırma",
    "🔍 Hiperparametre Analizi",
    "📚 Algoritma Teorisi"
])
 
# ═══════════════════════════════════════════════════
# SEKME 1 — MALİYET TAHMİNİ
# ═══════════════════════════════════════════════════
with t1:
    st.markdown("### 📈 Üretim Maliyeti Tahmini")
    st.markdown("Parametreleri ayarlayın — tahmin **anında** güncellenir.")
    st.divider()
 
    col_p, col_r = st.columns([1,2])
    with col_p:
        st.markdown("#### ⚙️ Parametreler")
        r_ur  = st.slider("📦 Üretim Miktarı (adet)", 50, 500, 250)
        r_mk  = st.slider("🔧 Makine Yaşı (yıl)", 1, 20, 8)
        r_is  = st.slider("👷 İşçilik Saati", 4.0, 16.0, 10.0, step=0.5)
        r_hm  = st.slider("🪨 Hammadde Fiyatı (TL/birim)", 10, 80, 45)
        r_vrd = st.radio("🌙 Vardiya", ["Gündüz","Gece"])
        r_vval = 1 if r_vrd=="Gündüz" else 2
        st.divider()
        model_r = st.radio("🤖 Model", ["Decision Tree","KNN","SVM"], horizontal=True)
 
    with col_r:
        girdi   = pd.DataFrame([{"uretim_miktari":r_ur,"makine_yasi":r_mk,
                                  "iscilik_saati":r_is,"hammadde_fiyati":r_hm,"vardiya":r_vval}])
        girdi_s = r_sc.transform(girdi)
        t_dt  = float(r_mdl["dt"].predict(girdi)[0])
        t_knn = float(r_mdl["knn"].predict(girdi_s)[0])
        t_svm = float(r_mdl["svm"].predict(girdi_s)[0])
        tahmin = {"Decision Tree":t_dt,"KNN":t_knn,"SVM":t_svm}[model_r]
 
        st.markdown(f"#### 💰 Tahmin — {model_r}")
        c1,c2 = st.columns(2)
        c1.metric("Toplam Maliyet", f"{tahmin:,.0f} TL")
        c2.metric("Birim Maliyet",  f"{tahmin/r_ur:,.0f} TL/adet")
        if tahmin > 25000:   st.error("⚠️ Yüksek maliyet")
        elif tahmin > 15000: st.warning("🟡 Orta maliyet")
        else:                st.success("✅ Normal maliyet")
        st.info(f"🌳 DT: **{t_dt:,.0f} TL** | 🔵 KNN: **{t_knn:,.0f} TL** | 🟣 SVM: **{t_svm:,.0f} TL**")
        st.divider()
        st.markdown("#### 📊 Test Seti Performansı")
        st.dataframe(pd.DataFrame({
            "Model": ["🌳 Decision Tree","🔵 KNN","🟣 SVM"],
            "R²":    [r_met["dt"]["r2"],r_met["knn"]["r2"],r_met["svm"]["r2"]],
            "MAE":   [f"{r_met['dt']['mae']:,.0f} TL",f"{r_met['knn']['mae']:,.0f} TL",f"{r_met['svm']['mae']:,.0f} TL"],
            "RMSE":  [f"{r_met['dt']['rmse']:,.0f} TL",f"{r_met['knn']['rmse']:,.0f} TL",f"{r_met['svm']['rmse']:,.0f} TL"],
        }), hide_index=True, use_container_width=True)
 
    st.divider()
    g1,g2 = st.columns(2)
    with g1:
        fig,ax = plt.subplots(figsize=(5,3.5))
        ax.scatter(yte_r, r_met["dt"]["pred"],  alpha=0.3,s=10,c="#e74c3c",label="DT")
        ax.scatter(yte_r, r_met["knn"]["pred"], alpha=0.3,s=10,c="#3498db",label="KNN")
        ax.scatter(yte_r, r_met["svm"]["pred"], alpha=0.3,s=10,c="#9b59b6",label="SVM")
        mn,mx = float(yte_r.min()),float(yte_r.max())
        ax.plot([mn,mx],[mn,mx],"k--",lw=1.5)
        ax.set_xlabel("Gerçek (TL)"); ax.set_ylabel("Tahmin (TL)")
        ax.set_title("Gerçek vs Tahmin"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig)
    with g2:
        imp = np.array(r_imp)
        lbl = ["Üretim","Makine Yaşı","İşçilik","Hammadde","Vardiya"]
        idx = np.argsort(imp)
        fig2,ax2 = plt.subplots(figsize=(5,3.5))
        ax2.barh([lbl[i] for i in idx], imp[idx],
                 color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"])
        for i,v in enumerate(imp[idx]):
            ax2.text(v+0.003,i,f"{v:.3f}",va="center",fontsize=9)
        ax2.set_title("Özellik Önemi — Decision Tree"); ax2.set_xlabel("Önem")
        plt.tight_layout(); st.pyplot(fig2)
 
# ═══════════════════════════════════════════════════
# SEKME 2 — ARIZA TAHMİNİ
# ═══════════════════════════════════════════════════
with t2:
    kaynak_lbl = {"uci":"AI4I 2020 (UCI)","yuklendi":"Yüklenen CSV","sentetik":"Demo — Sentetik"}
    st.markdown(f"### 🔴 Makine Arızası Tahmini — {kaynak_lbl.get(kaynak)}")
    st.divider()
 
    col_p2, col_r2 = st.columns([1,2])
    with col_p2:
        st.markdown("#### ⚙️ Sensör Değerleri")
        c_air  = st.slider("🌡️ Hava Sıcaklığı (K)", 295.0, 305.0, 300.0, step=0.1)
        c_proc = st.slider("🔥 İşlem Sıcaklığı (K)", 305.0, 315.0, 310.0, step=0.1)
        c_rpm  = st.slider("⚙️ Dönüş Hızı (rpm)", 1168, 2886, 1538)
        c_tork = st.slider("🔩 Tork (Nm)", 4.0, 77.0, 40.0, step=0.5)
        c_wear = st.slider("🔧 Takım Aşınması (dk)", 0, 250, 100)
        c_tip  = st.selectbox("🏷️ Ürün Tipi", ["L — Düşük","M — Orta","H — Yüksek"])
        c_te   = {"L — Düşük":0,"M — Orta":1,"H — Yüksek":2}[c_tip]
        st.divider()
        model_c = st.radio("🤖 Model", ["Decision Tree","KNN","SVM"], horizontal=True, key="mc")
 
    with col_r2:
        vals    = {"type_encoded":c_te,"air_temperature_k":c_air,
                   "process_temperature_k":c_proc,"rotational_speed_rpm":c_rpm,
                   "torque_nm":c_tork,"tool_wear_min":c_wear}
        grd     = pd.DataFrame([[vals.get(f,0) for f in c_feat]], columns=c_feat)
        grd_s   = c_sc.transform(grd)
 
        res = {}
        for k,m in c_mdl.items():
            g = grd if k=="dt" else grd_s
            res[k] = {"pred":int(m.predict(g)[0]),
                      "prob":float(m.predict_proba(g)[0][1])}
 
        aktif_key = {"Decision Tree":"dt","KNN":"knn","SVM":"svm"}[model_c]
        aktif = res[aktif_key]
 
        st.markdown(f"#### 🎯 Tahmin — {model_c}")
        if aktif["pred"]==1:
            st.error(f"🔴 **ARIZA RİSKİ YÜKSEK** — Olasılık: %{aktif['prob']*100:.1f}")
        else:
            st.success(f"✅ **NORMAL ÇALIŞMA** — Arıza olasılığı: %{aktif['prob']*100:.1f}")
        st.progress(float(aktif["prob"]))
 
        def lbl(r): return "🔴 ARIZA" if r["pred"]==1 else "✅ Normal"
        st.info(f"🌳 DT: **{lbl(res['dt'])}** (%{res['dt']['prob']*100:.1f})  |  "
                f"🔵 KNN: **{lbl(res['knn'])}** (%{res['knn']['prob']*100:.1f})  |  "
                f"🟣 SVM: **{lbl(res['svm'])}** (%{res['svm']['prob']*100:.1f})")
 
        st.divider()
        st.markdown("#### 📊 Test Seti Performansı")
        st.dataframe(pd.DataFrame({
            "Model":    ["🌳 DT","🔵 KNN","🟣 SVM"],
            "Accuracy": [f"%{c_met['dt']['acc']}",f"%{c_met['knn']['acc']}",f"%{c_met['svm']['acc']}"],
            "Recall ⭐":[f"%{c_met['dt']['rec']}",f"%{c_met['knn']['rec']}",f"%{c_met['svm']['rec']}"],
            "Precision":[f"%{c_met['dt']['prec']}",f"%{c_met['knn']['prec']}",f"%{c_met['svm']['prec']}"],
            "F1":       [f"%{c_met['dt']['f1']}",f"%{c_met['knn']['f1']}",f"%{c_met['svm']['f1']}"],
        }), hide_index=True, use_container_width=True)
        st.caption("⭐ Recall: gerçek arızaların kaçı tespit edildi — üretimde en kritik metrik")
 
    st.divider()
    st.markdown("#### Confusion Matrix")
    cm1,cm2,cm3 = st.columns(3)
    for col,key,name,cmap in [(cm1,"dt","DT","Reds"),(cm2,"knn","KNN","Blues"),(cm3,"svm","SVM","Purples")]:
        with col:
            p   = np.array(c_met[key]["pred"])
            cm  = confusion_matrix(yte_c, p)
            rec = c_met[key]["rec"]
            fig,ax = plt.subplots(figsize=(3,2.8))
            ax.imshow(cm, cmap=cmap)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Normal","Arıza"])
            ax.set_yticklabels(["Normal","Arıza"])
            ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
            ax.set_title(f"{name} — Recall %{rec}", fontsize=9, fontweight="bold")
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13,fontweight="bold",
                            color="white" if cm[i,j]>cm.max()/2 else "black")
            plt.tight_layout(); st.pyplot(fig)
 
# ═══════════════════════════════════════════════════
# SEKME 3 — KARŞILAŞTIRMA
# ═══════════════════════════════════════════════════
with t3:
    st.markdown("### 🆚 Model Karşılaştırması")
    st.divider()
    st.markdown("#### Regresyon")
    st.dataframe(pd.DataFrame({
        "Metrik":          ["R²","MAE (TL)","RMSE (TL)","Ölçeklendirme","Özellik Önemi"],
        "🌳 Decision Tree":[r_met["dt"]["r2"],f"{r_met['dt']['mae']:,.0f}",f"{r_met['dt']['rmse']:,.0f}","❌ Gerekmez","✅ Var"],
        "🔵 KNN":          [r_met["knn"]["r2"],f"{r_met['knn']['mae']:,.0f}",f"{r_met['knn']['rmse']:,.0f}","✅ Scaler","❌ Yok"],
        "🟣 SVM":          [r_met["svm"]["r2"],f"{r_met['svm']['mae']:,.0f}",f"{r_met['svm']['rmse']:,.0f}","✅ Scaler","❌ Yok"],
    }), hide_index=True, use_container_width=True)
 
    vals_r2 = [r_met["dt"]["r2"],r_met["knn"]["r2"],r_met["svm"]["r2"]]
    best_r2 = max(vals_r2)
    st.success(f"🏆 En yüksek R²: **{['Decision Tree','KNN','SVM'][vals_r2.index(best_r2)]}** ({best_r2})")
 
    st.divider()
    st.markdown("#### Sınıflandırma")
    st.dataframe(pd.DataFrame({
        "Metrik":          ["Accuracy","Recall ⭐","Precision","F1"],
        "🌳 Decision Tree":[f"%{c_met['dt']['acc']}",f"%{c_met['dt']['rec']}",f"%{c_met['dt']['prec']}",f"%{c_met['dt']['f1']}"],
        "🔵 KNN":          [f"%{c_met['knn']['acc']}",f"%{c_met['knn']['rec']}",f"%{c_met['knn']['prec']}",f"%{c_met['knn']['f1']}"],
        "🟣 SVM":          [f"%{c_met['svm']['acc']}",f"%{c_met['svm']['rec']}",f"%{c_met['svm']['prec']}",f"%{c_met['svm']['f1']}"],
    }), hide_index=True, use_container_width=True)
 
    vals_rec = [c_met["dt"]["rec"],c_met["knn"]["rec"],c_met["svm"]["rec"]]
    best_rec = max(vals_rec)
    st.success(f"🏆 En yüksek Recall: **{['Decision Tree','KNN','SVM'][vals_rec.index(best_rec)]}** (%{best_rec})")
 
    st.divider()
    st.dataframe(pd.DataFrame({
        "Özellik":    ["Paradigma","Eğitim","Tahmin Hızı","Ölçeklendirme","Yorumlanabilirlik","Hiperparametre"],
        "🌳 DT":      ["Eager/Kural","Ağaç öğrenir","⚡ O(log n)","❌ Gerekmez","⭐⭐⭐","max_depth"],
        "🔵 KNN":     ["Lazy/Örnek","Veriyi saklar","🐢 O(n·d)","✅ Zorunlu","⭐","K"],
        "🟣 SVM":     ["Eager/Marjin","SV bulur","⚡ O(sv·d)","✅ Zorunlu","⭐","C, kernel"],
    }), hide_index=True, use_container_width=True)
 
# ═══════════════════════════════════════════════════
# SEKME 4 — HİPERPARAMETRE
# ═══════════════════════════════════════════════════
with t4:
    st.markdown("### 🔍 Hiperparametre Optimizasyon Analizi")
    st.divider()
    hp = st.radio("Analiz:", ["KNN — K Optimizasyonu","SVM — C Optimizasyonu"], horizontal=True)
 
    @st.cache_data(show_spinner=False)
    def k_reg():
        df = uretim_verisi()
        X,y = df[r_feat],df["toplam_maliyet"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr); Xs,Xts = sc.transform(Xtr),sc.transform(Xte)
        tr,te=[],[]
        for k in range(1,21):
            m=KNeighborsRegressor(n_neighbors=k).fit(Xs,ytr)
            tr.append(r2_score(ytr,m.predict(Xs))); te.append(r2_score(yte,m.predict(Xts)))
        return tr,te
 
    @st.cache_data(show_spinner=False)
    def k_clf(vh):
        df = st.session_state.get("ai4i_df") or ai4i_sentetik()
        f = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                          "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
        X,y = df[f],df["machine_failure"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr); Xs,Xts = sc.transform(Xtr),sc.transform(Xte)
        tr,te=[],[]
        for k in range(1,21):
            m=KNeighborsClassifier(n_neighbors=k).fit(Xs,ytr)
            tr.append(accuracy_score(ytr,m.predict(Xs))); te.append(accuracy_score(yte,m.predict(Xts)))
        return tr,te
 
    @st.cache_data(show_spinner=False)
    def c_reg():
        df = uretim_verisi()
        X,y = df[r_feat],df["toplam_maliyet"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr); Xs,Xts = sc.transform(Xtr),sc.transform(Xte)
        cv=[0.01,0.1,1,10,50,100,200,500,1000]; tr,te=[],[]
        for c in cv:
            m=SVR(kernel="rbf",C=c).fit(Xs,ytr)
            tr.append(r2_score(ytr,m.predict(Xs))); te.append(r2_score(yte,m.predict(Xts)))
        return cv,tr,te
 
    @st.cache_data(show_spinner=False)
    def c_clf(vh):
        df = st.session_state.get("ai4i_df") or ai4i_sentetik()
        f = [c for c in ["type_encoded","air_temperature_k","process_temperature_k",
                          "rotational_speed_rpm","torque_nm","tool_wear_min"] if c in df.columns]
        X,y = df[f],df["machine_failure"]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc = StandardScaler().fit(Xtr); Xs,Xts = sc.transform(Xtr),sc.transform(Xte)
        cv=[0.01,0.1,1,10,50,100,200,500,1000]; tr,te=[],[]
        for c in cv:
            m=SVC(kernel="rbf",C=c,random_state=42).fit(Xs,ytr)
            tr.append(accuracy_score(ytr,m.predict(Xs))); te.append(accuracy_score(yte,m.predict(Xts)))
        return cv,tr,te
 
    def plot_hp(ax, x, tr, te, xlabel, ylabel, title, used_val):
        ax.plot(range(len(x)),tr,"o-",c="#e74c3c",lw=2,label="Eğitim")
        ax.plot(range(len(x)),te,"s-",c="#3498db",lw=2,label="Test")
        bi = int(np.argmax(te))
        ax.axvline(bi,c="green",ls="--",lw=1.5,label=f"En iyi={x[bi]}")
        if used_val in x:
            ax.axvline(x.index(used_val),c="gray",ls=":",lw=1,label=f"Kullanılan={used_val}")
        ax.set_xticks(range(len(x))); ax.set_xticklabels([str(v) for v in x],rotation=45,fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title,fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
    if "KNN" in hp:
        with st.spinner("Hesaplanıyor..."):
            tr_r,te_r = k_reg()
            tr_c,te_c = k_clf(veri_hash)
        h1,h2 = st.columns(2)
        with h1:
            fig,ax=plt.subplots(figsize=(5,3.5))
            plot_hp(ax,list(range(1,21)),tr_r,te_r,"K","R²","Regresyon — K vs R²",5)
            plt.tight_layout(); st.pyplot(fig)
        with h2:
            fig,ax=plt.subplots(figsize=(5,3.5))
            plot_hp(ax,list(range(1,21)),tr_c,te_c,"K","Accuracy","Sınıflandırma — K vs Accuracy",5)
            plt.tight_layout(); st.pyplot(fig)
    else:
        with st.spinner("Hesaplanıyor..."):
            cv,tr_r,te_r = c_reg()
            _,tr_c,te_c  = c_clf(veri_hash)
        h1,h2 = st.columns(2)
        with h1:
            fig,ax=plt.subplots(figsize=(5,3.5))
            plot_hp(ax,cv,tr_r,te_r,"C","R²","SVR — C vs R²",100)
            plt.tight_layout(); st.pyplot(fig)
        with h2:
            fig,ax=plt.subplots(figsize=(5,3.5))
            plot_hp(ax,cv,tr_c,te_c,"C","Accuracy","SVC — C vs Accuracy",1)
            plt.tight_layout(); st.pyplot(fig)
 
    st.divider()
    st.dataframe(pd.DataFrame({
        "Durum":   ["Overfitting","Underfitting","Optimal"],
        "🌳 DT":   ["max_depth büyük","max_depth küçük","Test kaybı minimum"],
        "🔵 KNN":  ["K=1","K çok büyük","Test R²/Acc maksimum"],
        "🟣 SVM":  ["C çok büyük","C çok küçük","Test performansı maksimum"],
    }), hide_index=True, use_container_width=True)
 
# ═══════════════════════════════════════════════════
# SEKME 5 — TEORİ
# ═══════════════════════════════════════════════════
with t5:
    st.markdown("### 📚 Algoritma Teorisi")
    st.divider()
    a1,a2,a3 = st.tabs(["🌳 Decision Tree","🔵 KNN","🟣 SVM"])
    with a1:
        st.markdown("""
**Eager Learning — Kural Tabanlı**
 
Veriyi işler, kalıcı bir ağaç modeli öğrenir.
- **Regresyon:** MSE minimize → yaprak ortalaması
- **Sınıflandırma:** Gini indeksi → çoğunluk sınıfı
- Ölçeklendirme **gerekmez** — kararlar sıralama ilişkisine dayanır
- Özellik önemi **hesaplanabilir**
 
`max_depth` küçük → Underfitting | `max_depth` büyük → Overfitting
        """)
    with a2:
        st.markdown("""
**Lazy Learning — Örnek Tabanlı**
 
Eğitim aşamasında sadece veriyi saklar, tahmin anında mesafe hesaplar.
- **Regresyon:** K komşunun ortalaması
- **Sınıflandırma:** K komşunun çoğunluk oyu
- Ölçeklendirme **ZORUNLU** — farklı birimler mesafeyi bozar
  `z = (x − μ) / σ`
 
`K=1` → Overfitting | `K büyük` → Underfitting
        """)
    with a3:
        st.markdown("""
**Eager Learning — Marjin Tabanlı**
 
Sınıflar arasındaki **maksimum marjini** bulan hiper düzlem.
- **C:** küçük = geniş marjin | büyük = dar marjin (overfitting riski)
- **RBF Kernel:** doğrusal olmayan sınırları modeller
- **SVR:** ε-duyarsız tüp — tüp içindeki hatalar cezalandırılmaz
- Ölçeklendirme **ZORUNLU**
 
`C küçük` → Underfitting | `C büyük` → Overfitting
        """)
    st.divider()
    st.dataframe(pd.DataFrame({
        "Kriter":  ["Yorumlanabilirlik şart","Gerçek zamanlı tahmin","Büyük veri","Az veri","Doğrusal olmayan"],
        "Öneri":   ["🌳 DT","🌳 DT veya 🟣 SVM","🌳 DT veya 🟣 SVM","🟣 SVM","🟣 SVM (RBF)"],
    }), hide_index=True, use_container_width=True)
 
