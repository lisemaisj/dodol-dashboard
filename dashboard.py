import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dodol Production Monitor",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Compact CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding: 0.6rem 1.2rem 0.5rem 1.2rem !important; }
    div[data-testid="metric-container"] { padding: 4px 8px !important; }
    div[data-testid="metric-container"] label { font-size: 0.72rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.68rem !important; }
    .stNumberInput label { font-size: 0.78rem !important; }
    .stNumberInput input  { padding: 4px 8px !important; font-size: 0.85rem !important; }
    .stButton button { padding: 6px 0 !important; font-size: 0.85rem !important; }
    h1 { font-size: 1.2rem !important; margin-bottom: 0 !important; padding-bottom: 0 !important; }
    h2, h3 { font-size: 0.95rem !important; margin: 4px 0 2px 0 !important; }
    .stAlert  { padding: 6px 10px !important; font-size: 0.78rem !important; }
    .stCaption { font-size: 0.7rem !important; margin: 0 !important; }
    hr { margin: 6px 0 !important; }
    .stMarkdown p { margin: 2px 0 !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TEMP_TARGET   = 114.0
TEMP_SD       = 1.2
TEMP_USL      = 118.0
TEMP_LSL      = 108.0
VISC_PASS_MIN = 19000
VISC_PASS_MAX = 24000
LAMBDA_EWMA   = 0.2
L_SIGMA       = 3
UCL = TEMP_TARGET + L_SIGMA * TEMP_SD * np.sqrt(LAMBDA_EWMA / (2 - LAMBDA_EWMA))
LCL = TEMP_TARGET - L_SIGMA * TEMP_SD * np.sqrt(LAMBDA_EWMA / (2 - LAMBDA_EWMA))

SPC_FEATURES = ['EWMA','MA_2','Dist_CL','Process_SD',
                'UCL_Viol','LCL_Viol','Run_Length','Cp','Cpk','CookingHour']

PARAM_RANGES = {
    'Temperature_C':  {'min':100.0,'max':130.0,'target':114.0,'unit':'°C'},  # wide — spec is 108–118
    'MotorCurrent_A': {'min':  0.0,'max': 25.0,'target': 11.0,'unit':'A'},   # wide — spec is 9–14
    'CookingTime_hr': {'min':    1,'max':    6,'target':    3,'unit':'hr'},
}

# Spec limits shown on gauges (for visual reference only — not input constraints)
SPEC_LIMITS = {
    'Temperature_C':  {'lsl':108.0, 'usl':118.0},
    'MotorCurrent_A': {'lsl':  9.0, 'usl': 14.0},
}


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    missing = [p for p in ['svm_model.pkl','scaler.pkl'] if not os.path.exists(p)]
    if missing:
        return None, None, False, missing
    return joblib.load('svm_model.pkl'), joblib.load('scaler.pkl'), True, []

svm_model, scaler, model_ok, missing_files = load_model()
if not model_ok:
    st.error("❌ Missing: " + ", ".join(f"`{f}`" for f in missing_files))
    st.stop()


# ── Compact risk chart ────────────────────────────────────────────────────────
def risk_chart(prob_fail_history, ewma_history):
    hrs  = list(range(1, len(prob_fail_history) + 1))
    pcts = [p * 100 for p in prob_fail_history]

    fig, ax1 = plt.subplots(figsize=(11, 2.4))
    fig.patch.set_facecolor('#0e1117'); ax1.set_facecolor('#161b22')

    colours = ['#d62728' if p >= 35 else '#ff7f0e' if p >= 22 else '#2ca02c' for p in pcts]
    bars = ax1.bar(hrs, pcts, color=colours, alpha=0.8, width=0.55, zorder=2)
    for bar, val in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                 f'{val:.0f}%', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold', color='white')
    ax1.axhspan(0,  22, alpha=0.05, color='#2ca02c')
    ax1.axhspan(22, 35, alpha=0.05, color='#ff7f0e')
    ax1.axhspan(35,100, alpha=0.05, color='#d62728')
    ax1.axhline(22, color='#ff7f0e', linewidth=0.7, linestyle=':', alpha=0.5)
    ax1.axhline(35, color='#d62728', linewidth=0.7, linestyle=':', alpha=0.5)
    ax1.set_ylabel('Fail Risk %', color='#aaaaaa', fontsize=7.5)
    ax1.set_ylim(0, 108); ax1.set_xlabel('Cooking Hour', color='#aaaaaa', fontsize=7.5)
    ax1.set_xticks(hrs); ax1.tick_params(colors='#888888', labelsize=7)
    ax1.spines[:].set_color('#333333')

    ax2 = ax1.twinx()
    ax2.plot(hrs, ewma_history, color='#4a90d9', linewidth=1.8, marker='o', markersize=4, zorder=3)
    ucl_mask = np.array(ewma_history) > UCL
    lcl_mask = np.array(ewma_history) < LCL
    if ucl_mask.any():
        ax2.scatter(np.array(hrs)[ucl_mask], np.array(ewma_history)[ucl_mask],
                    color='#ff4444', s=45, zorder=5)
    if lcl_mask.any():
        ax2.scatter(np.array(hrs)[lcl_mask], np.array(ewma_history)[lcl_mask],
                    color='#ffaa00', s=45, zorder=5)
    ax2.axhline(UCL, color='#ff4444', linewidth=1, linestyle='--', alpha=0.6, label=f'Upper {UCL:.1f}°C')
    ax2.axhline(LCL, color='#ffaa00', linewidth=1, linestyle='--', alpha=0.6, label=f'Lower {LCL:.1f}°C')
    ax2.axhline(TEMP_TARGET, color='#2ca02c', linewidth=0.9, linestyle=':', alpha=0.5, label=f'Target {TEMP_TARGET}°C')
    ax2.set_ylabel('Temp (°C)', color='#4a90d9', fontsize=7.5)
    ax2.tick_params(colors='#4a90d9', labelsize=7); ax2.spines[:].set_color('#333333')
    ax2.legend(fontsize=6.5, loc='upper right', facecolor='#161b22',
               labelcolor='white', framealpha=0.8, ncol=3)
    ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
    plt.tight_layout(pad=0.4)
    return fig


# ── Compact gauge ─────────────────────────────────────────────────────────────
def gauge_chart(value, param_key, title):
    r    = PARAM_RANGES[param_key]
    pct  = max(0, min(1, (value - r['min']) / max(r['max'] - r['min'], 1e-9)))
    spec = SPEC_LIMITS.get(param_key)

    # Colour: red if outside spec, orange if near spec edge, green if near target
    if spec and (value < spec['lsl'] or value > spec['usl']):
        col = '#d62728'  # outside spec
    elif spec and (value < spec['lsl'] + 1.5 or value > spec['usl'] - 1.5):
        col = '#ff7f0e'  # near spec edge
    else:
        dist = abs(value - r['target'])
        span = (r['max'] - r['min']) / 2
        col  = '#2ca02c' if dist < span*0.3 else '#ff7f0e' if dist < span*0.7 else '#d62728'

    fig, ax = plt.subplots(figsize=(2.6, 1.7), subplot_kw={'projection':'polar'})
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
    ax.plot(np.linspace(np.pi,0,200), [1]*200, linewidth=10, color='#2a2a2a', solid_capstyle='round')
    ax.plot(np.linspace(np.pi, np.pi-pct*np.pi, 200), [1]*200, linewidth=10, color=col, solid_capstyle='round')

    # Target marker (white)
    tgt_theta = np.pi - ((r['target']-r['min'])/(r['max']-r['min']))*np.pi
    ax.plot([tgt_theta,tgt_theta],[0.78,1.12], color='white', linewidth=1.5)

    # Spec limit markers (red/yellow ticks) if available
    if spec:
        lsl_theta = np.pi - ((spec['lsl']-r['min'])/(r['max']-r['min']))*np.pi
        usl_theta = np.pi - ((spec['usl']-r['min'])/(r['max']-r['min']))*np.pi
        ax.plot([lsl_theta,lsl_theta],[0.82,1.08], color='#ffaa00', linewidth=1.2)
        ax.plot([usl_theta,usl_theta],[0.82,1.08], color='#ff4444', linewidth=1.2)

    ax.set_ylim(0,1.3); ax.axis('off')
    ax.text(0,-0.1,f"{value:.1f}{r['unit']}", ha='center', va='center',
            fontsize=11, fontweight='bold', color=col, transform=ax.transAxes)
    ax.text(0,-0.32, title, ha='center', va='center',
            fontsize=7.5, color='#aaaaaa', transform=ax.transAxes)
    # Show out-of-spec warning text
    if spec and (value < spec['lsl'] or value > spec['usl']):
        ax.text(0,-0.52, '⚠ Out of spec', ha='center', va='center',
                fontsize=7, color='#d62728', transform=ax.transAxes)
    else:
        ax.text(0,-0.52, f"Spec: {spec['lsl']}–{spec['usl']}{r['unit']}" if spec else '',
                ha='center', va='center', fontsize=6.5, color='#555555', transform=ax.transAxes)
    plt.tight_layout(pad=0)
    return fig


# ── Batch progress chart (sidebar) ───────────────────────────────────────────
def batch_progress_chart(ewma_history, temp_history):
    hrs = list(range(1, len(ewma_history)+1))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#161b22')
    ax.plot(hrs, temp_history, color='#4a90d9', alpha=0.3, linewidth=1, linestyle='--')
    ax.plot(hrs, ewma_history, color='#4a90d9', linewidth=2, marker='o', markersize=4, label='Smoothed Temp')
    ucl_mask = np.array(ewma_history) > UCL
    lcl_mask = np.array(ewma_history) < LCL
    if ucl_mask.any():
        ax.scatter(np.array(hrs)[ucl_mask], np.array(ewma_history)[ucl_mask], color='#ff4444', s=55, zorder=5, label='🔴 Too Hot')
    if lcl_mask.any():
        ax.scatter(np.array(hrs)[lcl_mask], np.array(ewma_history)[lcl_mask], color='#ffaa00', s=55, zorder=5, label='🟡 Too Cold')
    ax.axhline(UCL, color='#ff4444', linewidth=1.2, linestyle='--', label=f'Upper {UCL:.1f}°C')
    ax.axhline(LCL, color='#ffaa00', linewidth=1.2, linestyle='--', label=f'Lower {LCL:.1f}°C')
    ax.axhline(TEMP_TARGET, color='#2ca02c', linewidth=1, linestyle=':', label=f'Target {TEMP_TARGET}°C')
    ax.set_xlabel('Cooking Hour', color='#cccccc', fontsize=8)
    ax.set_ylabel('Temperature (°C)', color='#cccccc', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=7.5); ax.spines[:].set_color('#333333')
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='white', ncol=3)
    plt.tight_layout(pad=0.4)
    return fig


# ── SPC deviation chart (sidebar) ─────────────────────────────────────────────
def spc_deviation_chart(spc_values):
    arr      = np.array(spc_values)
    z_scores = np.abs((arr - scaler.mean_) / (scaler.scale_ + 1e-9))
    labels   = ['Smoothed Temp','Avg Temp (2hr)','Drift','Consistency',
                'Above Limit','Below Limit','Hrs OK','Spread','Centring','Hour']
    colours  = ['#d62728' if z == z_scores.max() else '#4393c3' for z in z_scores]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#161b22')
    bars = ax.barh(labels, z_scores, color=colours, edgecolor='#0e1117', height=0.55)
    for bar, val in zip(bars, z_scores):
        ax.text(val+0.02, bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=7.5, fontweight='bold', color='white')
    ax.axvline(1.0, color='#ffaa00', linewidth=1, linestyle='--')
    ax.axvline(2.0, color='#ff4444', linewidth=1, linestyle='--')
    ax.set_xlabel('Deviation from normal', color='#cccccc', fontsize=7.5)
    ax.tick_params(colors='#aaaaaa', labelsize=7); ax.spines[:].set_color('#333333')
    plt.tight_layout(pad=0.4)
    return fig, labels[int(np.argmax(z_scores))]


# ── Early warning engine ──────────────────────────────────────────────────────
def evaluate_early_warnings(ewma_history, prob_fail_history, ucl_viol_log, lcl_viol_log):
    out = []
    n   = len(ewma_history)

    # W1 — Sustained drift
    if n >= 3:
        last3 = ewma_history[-3:]
        diffs = [last3[i+1]-last3[i] for i in range(2)]
        if all(d > 0 for d in diffs) and last3[-1] > TEMP_TARGET:
            out.append({'level':'warning','triggered':True,
                'rule':'↗ Rising 3 hrs', 'message':f'Now {last3[-1]:.1f}°C — limit {UCL:.1f}°C. Reduce heat.'})
        elif all(d < 0 for d in diffs) and last3[-1] < TEMP_TARGET:
            out.append({'level':'warning','triggered':True,
                'rule':'↘ Falling 3 hrs', 'message':f'Now {last3[-1]:.1f}°C — limit {LCL:.1f}°C. Check heat.'})
        else:
            out.append({'level':'ok','triggered':False,'rule':'✅ Trend','message':'Stable'})
    else:
        out.append({'level':'info','triggered':False,'rule':'ℹ Trend','message':'Need 3 hrs'})

    # W2 — Consecutive violations
    if n >= 2:
        if sum(1 for v in ucl_viol_log[-2:] if v==1) >= 2:
            out.append({'level':'error','triggered':True,
                'rule':'🔴 Too Hot × 2', 'message':f'Above {UCL:.1f}°C twice. Check cooker.'})
        elif sum(1 for v in lcl_viol_log[-2:] if v==1) >= 2:
            out.append({'level':'error','triggered':True,
                'rule':'🔴 Too Cold × 2', 'message':f'Below {LCL:.1f}°C twice. Check heat.'})
        else:
            out.append({'level':'ok','triggered':False,'rule':'✅ Limits','message':'No repeat violations'})
    else:
        out.append({'level':'info','triggered':False,'rule':'ℹ Limits','message':'Need 2 hrs'})

    # W3 — Rising fail risk
    if len(prob_fail_history) >= 3:
        chg = prob_fail_history[-1] - prob_fail_history[-3]
        if chg >= 0.15:
            out.append({'level':'error','triggered':True,
                'rule':'📈 Risk +Fast', 'message':f'+{chg*100:.0f}% in 3 hrs ({prob_fail_history[-3]*100:.0f}%→{prob_fail_history[-1]*100:.0f}%)'})
        elif chg >= 0.08:
            out.append({'level':'warning','triggered':True,
                'rule':'📈 Risk Creeping', 'message':f'+{chg*100:.0f}% in 3 hrs. Watch next hr.'})
        else:
            out.append({'level':'ok','triggered':False,'rule':'✅ Risk','message':f'Stable {prob_fail_history[-1]*100:.0f}%'})
    elif len(prob_fail_history) >= 2:
        chg = prob_fail_history[-1] - prob_fail_history[-2]
        if chg >= 0.15:
            out.append({'level':'warning','triggered':True,
                'rule':'📈 Risk Jumped', 'message':f'+{chg*100:.0f}% this hr. Watch next.'})
        else:
            out.append({'level':'ok','triggered':False,'rule':'✅ Risk','message':'Stable'})
    else:
        out.append({'level':'info','triggered':False,'rule':'ℹ Risk','message':'Need 2 hrs'})

    # W4 — Near limit
    if n >= 1:
        cur  = ewma_history[-1]
        d_ucl = UCL - cur; d_lcl = cur - LCL
        if 0 < d_ucl < 0.5:
            out.append({'level':'warning','triggered':True,
                'rule':'⚠ Near Upper', 'message':f'{d_ucl:.2f}°C to limit. Reduce heat slightly.'})
        elif 0 < d_lcl < 0.5:
            out.append({'level':'warning','triggered':True,
                'rule':'⚠ Near Lower', 'message':f'{d_lcl:.2f}°C to limit. Check heat.'})
        else:
            out.append({'level':'ok','triggered':False,'rule':'✅ Proximity',
                'message':f'{UCL-cur:.1f}°C to upper · {cur-LCL:.1f}°C to lower'})
    return out


# ── Session state ─────────────────────────────────────────────────────────────
for k, d in [('ewma_val',None),('ewma_history',[]),('temp_history',[]),
              ('reading_log',[]),('latest',None),
              ('prob_fail_history',[]),('ucl_viol_log',[]),('lcl_viol_log',[])]:
    if k not in st.session_state:
        st.session_state[k] = d


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.success("✅ Model ready")
    st.caption(f"🎯 Target: {TEMP_TARGET}°C  |  Range: {TEMP_LSL}–{TEMP_USL}°C")
    st.caption(f"🔴 Upper: {UCL:.2f}°C  |  🟡 Lower: {LCL:.2f}°C")
    if st.button("🔄 Start New Batch", use_container_width=True):
        for k in ['ewma_val','ewma_history','temp_history','reading_log',
                  'latest','prob_fail_history','ucl_viol_log','lcl_viol_log']:
            st.session_state[k] = None if k in ('ewma_val','latest') else []
        st.rerun()
    st.markdown("---")

    # Batch progress (sidebar)
    if st.session_state['ewma_history']:
        st.markdown("**📈 Batch Progress**")
        fig_bp = batch_progress_chart(st.session_state['ewma_history'], st.session_state['temp_history'])
        st.pyplot(fig_bp, use_container_width=True); plt.close()

    # Gauges (sidebar)
    if st.session_state['latest']:
        L = st.session_state['latest']
        st.markdown("**📡 Gauges**")
        gc1, gc2, gc3 = st.columns(3)
        with gc1: st.pyplot(gauge_chart(L['live_temp'],    'Temperature_C',  '🌡 Temp'), use_container_width=True); plt.close()
        with gc2: st.pyplot(gauge_chart(L['live_current'], 'MotorCurrent_A', '⚡ Amp'),  use_container_width=True); plt.close()
        with gc3: st.pyplot(gauge_chart(float(L['live_hour']), 'CookingTime_hr','⏱ Hr'), use_container_width=True); plt.close()
        st.caption("⚡ Motor current is for display only.")

    # Reading log (sidebar)
    if st.session_state['reading_log']:
        st.markdown("**📋 Batch Log**")
        log_df = pd.DataFrame(st.session_state['reading_log'])
        def hl(val):
            return 'color:#2ca02c;font-weight:bold' if val=='Pass' else 'color:#d62728;font-weight:bold'
        st.dataframe(log_df.style.map(hl, subset=['Outcome']),
                     height=200, use_container_width=True)

    # Supervisor section
    if st.session_state['latest']:
        with st.expander("🔧 Supervisor Details"):
            fig_dev, top_feat = spc_deviation_chart(st.session_state['latest']['spc_values'])
            st.pyplot(fig_dev, use_container_width=True); plt.close()
            st.caption(f"Most deviated: **{top_feat}**")
            st.caption(f"Model: svm_B (trained on historical data) | λ={LAMBDA_EWMA} | σ={TEMP_SD}°C")


# ── Main layout: header ───────────────────────────────────────────────────────
st.markdown("### 🍬 Dodol Batch Quality Monitor")
st.caption("Enter process readings after each cooking hour. The AI model — trained on historical production data — predicts batch quality based on learned patterns.")
st.markdown("---")

# ── Main layout: LEFT (input) | RIGHT (prediction + alerts) ──────────────────
left, right = st.columns([1, 2], gap="medium")

with left:
    st.markdown("**📥 Enter Readings**")
    live_temp = st.number_input("🌡 Temperature (°C)",
        value=114.0, step=0.1, format="%.1f",
        help="Target 114°C · Spec 108–118°C · No hard limit — enter actual reading")
    live_current = st.number_input("⚡ Motor Current (A)",
        value=11.0, step=0.05, format="%.2f",
        help="Display only — does not affect prediction")
    live_hour = st.number_input("⏱ Cooking Hour",
        min_value=1, max_value=6,
        value=min(max(1, len(st.session_state['reading_log'])+1), 6), step=1)
    submit = st.button("✅ Record This Hour", type="primary", use_container_width=True)

    if submit:
        temps  = st.session_state['temp_history']
        n_before = len(temps)

        # EWMA cold-start matches pandas ewm(adjust=False)
        new_ewma = live_temp if n_before == 0 else \
                   LAMBDA_EWMA * live_temp + (1 - LAMBDA_EWMA) * st.session_state['ewma_val']
        st.session_state['ewma_val'] = new_ewma
        st.session_state['ewma_history'].append(new_ewma)
        st.session_state['temp_history'].append(live_temp)

        temps = st.session_state['temp_history']
        ewmas = st.session_state['ewma_history']
        n     = len(temps)

        dist_cl  = abs(new_ewma - TEMP_TARGET)
        ucl_viol = float(new_ewma > UCL)
        lcl_viol = float(new_ewma < LCL)
        ma2      = (live_temp + temps[-2]) / 2 if n >= 2 else live_temp
        proc_sd  = float(np.std(temps[-3:], ddof=1)) if n >= 2 else TEMP_SD
        proc_sd  = proc_sd if proc_sd > 0 else TEMP_SD

        run_len = 0
        for v in reversed(ewmas):
            if LCL <= v <= UCL: run_len += 1
            else: break

        batch_mean = float(np.mean(temps))
        batch_sd   = float(np.std(temps, ddof=1)) if n >= 2 else TEMP_SD
        batch_sd   = batch_sd if batch_sd > 0 else TEMP_SD
        cp  = (TEMP_USL - TEMP_LSL) / (6 * batch_sd)
        cpk = min((TEMP_USL - batch_mean) / (3 * batch_sd),
                  (batch_mean - TEMP_LSL)  / (3 * batch_sd))

        spc_values = [new_ewma, ma2, dist_cl, proc_sd,
                      ucl_viol, lcl_viol, float(run_len),
                      cp, cpk, float(live_hour)]

        probs      = svm_model.predict_proba(scaler.transform([spc_values]))[0]
        prob_pass  = float(probs[1])
        prob_fail  = float(probs[0])
        prediction = 'Pass' if prob_fail < 0.22 else 'Fail'  # threshold tuned from P75 of Pass-row P(Fail) distribution

        st.session_state['prob_fail_history'].append(prob_fail)
        st.session_state['ucl_viol_log'].append(int(ucl_viol))
        st.session_state['lcl_viol_log'].append(int(lcl_viol))

        st.session_state['reading_log'].append({
            'Hr':        live_hour,
            'Temp °C':   round(live_temp, 1),
            'Current A': round(live_current, 2),
            'Smooth °C': round(new_ewma, 2),
            '🔴 Hi':     int(ucl_viol),
            '🟡 Lo':     int(lcl_viol),
            '✅ Hrs OK': run_len,
            'Risk %':    round(prob_fail * 100, 1),
            'Outcome':   prediction,
        })

        st.session_state['latest'] = dict(
            prob_pass=prob_pass, prob_fail=prob_fail, prediction=prediction,
            new_ewma=new_ewma, ucl_viol=ucl_viol, lcl_viol=lcl_viol,
            cp=cp, cpk=cpk, spc_values=spc_values,
            live_temp=live_temp, live_current=live_current,
            live_hour=live_hour, n_readings=n,
        )
        st.rerun()

    # Mini status strip
    if st.session_state['latest']:
        L = st.session_state['latest']
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Smoothed Temp", f"{L['new_ewma']:.1f}°C",
                  delta=f"{L['new_ewma']-TEMP_TARGET:+.1f}°C")
        m2.metric("Hrs Recorded", str(L['n_readings']))
        m3, m4 = st.columns(2)
        m3.metric("Spread (Cp)",   f"{L['cp']:.2f}",
                  delta="✅" if L['cp']  >= 1.33 else "⚠",
                  delta_color="normal" if L['cp']  >= 1.33 else "inverse")
        m4.metric("Centring (Cpk)", f"{L['cpk']:.2f}",
                  delta="✅" if L['cpk'] >= 1.0  else "⚠",
                  delta_color="normal" if L['cpk'] >= 1.0  else "inverse")


with right:
    if st.session_state['latest']:
        L = st.session_state['latest']
        prob_pass  = L['prob_pass']; prob_fail = L['prob_fail']
        prediction = L['prediction']; new_ewma = L['new_ewma']
        ucl_viol   = L['ucl_viol'];  lcl_viol  = L['lcl_viol']
        n_readings = L['n_readings']

        pc = "#2ca02c" if prediction=='Pass' else "#d62728"
        rl = "🟢 Low Risk" if prob_fail<0.22 else "🟡 Medium Risk" if prob_fail<0.35 else "🔴 High Risk"

        # Action guidance
        if prediction == 'Pass' and prob_fail < 0.22:
            action_icon, action_title, action_colour = "🟢", "Outlook: Batch on track", "#2ca02c"
            action_steps = [
                "Batch is predicted to pass quality check.",
                "Continue cooking as normal.",
                "Record next hour's readings when due.",
            ]
        elif prediction == 'Pass' and prob_fail < 0.35:
            action_icon, action_title, action_colour = "🟡", "Outlook: Proceed with caution", "#ff9f1c"
            action_steps = [
                "Batch is predicted to pass — but risk is slightly elevated.",
                "Monitor temperature closely over the next hour.",
                "Ensure heat is stable and not drifting toward the limit.",
                "Do not leave the cooker unattended.",
            ]
        elif prediction == 'Pass' and prob_fail >= 0.35:
            action_icon, action_title, action_colour = "🟠", "Outlook: High risk — stay alert", "#ff7f0e"
            action_steps = [
                "Batch is still predicted to pass — but current readings show elevated risk.",
                "Verify temperature sensor is reading correctly.",
                "Check heat source stability and stirring speed.",
                "If conditions do not improve next hour, escalate to supervisor.",
            ]
        else:
            action_icon, action_title, action_colour = "🔴", "Outlook: Action required", "#d62728"
            action_steps = [
                "Based on current readings, this batch is predicted to fail quality check.",
                "Verify temperature is within spec (108–118°C) to prevent further deviation.",
                "Check motor current for signs of stirring inconsistency.",
                "Correct heat setting now — the next hour's reading will update the prediction.",
                "Inform supervisor if process cannot be stabilised.",
            ]

        # Prediction card + temp status side by side
        card_col, temp_col = st.columns([3, 2])

        with card_col:
            steps_html = "".join(
                f"<div style='margin:3px 0;font-size:0.76em;color:#dddddd;'>▶ {s}</div>"
                for s in action_steps
            )
            card_html = f"""
            <div style='font-family: "Source Sans 3", "Source Sans Pro", "Helvetica Neue", Arial, sans-serif; font-size:0.85rem; line-height:1.45; color:#dddddd; background:{pc}18; border:2px solid {pc}; border-radius:12px;
                        padding:12px 16px;'>
                <div style='font-size:0.72em;color:#aaaaaa;'>
                    AI Quality Model &nbsp;|&nbsp; Based on {n_readings} hr(s) of readings
                </div>
                <div style='font-size:0.75em;color:#aaaaaa;margin-top:4px;letter-spacing:0.05em;'>
                    BATCH IS PREDICTED TO:
                </div>
                <div style='font-size:2.2em;font-weight:bold;color:{pc};margin:2px 0;'>
                    {'✅ Pass' if prediction=='Pass' else '❌ Fail'}
                </div>
                <div style='font-size:0.7em;color:#888888;margin-bottom:6px;font-style:italic;'>
                    Prediction based on learned historical patterns · confidence improves with more readings
                </div>
                {f'<span style="background:{action_colour};color:white;border-radius:16px;padding:2px 12px;font-size:0.82em;font-weight:bold;">{rl}</span>' if prediction == "Pass" and prob_fail >= 0.15 else ""}
                <div style='margin-top:8px;font-size:0.85em;color:#cccccc;'>
                    🟢 Pass: <b>{prob_pass*100:.0f}%</b>
                    &nbsp;&nbsp;
                    🔴 Fail: <b>{prob_fail*100:.0f}%</b>
                </div>
                <div style='margin-top:10px;padding-top:8px;
                            border-top:1px solid {action_colour}55;'>
                    <div style='font-size:0.8em;font-weight:bold;
                                color:{action_colour};margin-bottom:4px;'>
                        {action_icon} {action_title}
                    </div>
                    {steps_html}
                </div>
            </div>
            """
            components.html(card_html, height=360)

        with temp_col:
            n_viol = sum(1 for r in st.session_state['reading_log']
                         if r['🔴 Hi']==1 or r['🟡 Lo']==1)
            st.metric("🌡 Smoothed Temp", f"{new_ewma:.1f}°C",
                      delta=f"{new_ewma-TEMP_TARGET:+.1f}°C vs target")
            st.metric("⚠ Hrs Out of Range", str(n_viol),
                      delta=f"{n_viol/n_readings*100:.0f}%", delta_color="inverse")
            if   ucl_viol: st.error("🔴 Too Hot Now")
            elif lcl_viol: st.warning("🟡 Too Cold Now")
            else:          st.success("🟢 In Range")

        # Alerts
        st.markdown("**🚨 Alerts**")
        ew = evaluate_early_warnings(
            st.session_state['ewma_history'], st.session_state['prob_fail_history'],
            st.session_state['ucl_viol_log'], st.session_state['lcl_viol_log'])
        n_active = sum(1 for w in ew if w['triggered'])
        if n_active == 0:
            st.success("✅ All clear")
        a1, a2 = st.columns(2)
        for i, w in enumerate(ew):
            col = a1 if i % 2 == 0 else a2
            msg = f"**{w['rule']}** — {w['message']}"
            if   w['level']=='error'   and w['triggered']: col.error(msg)
            elif w['level']=='warning' and w['triggered']: col.warning(msg)
            elif w['level']=='ok':                         col.success(msg)
            else:                                          col.info(msg)
    else:
        st.info("👈 Enter this hour's process readings and click **✅ Record This Hour**. The AI model will predict batch quality based on patterns learned from historical production data.")

# ── Bottom: Risk chart ────────────────────────────────────────────────────────
if st.session_state['prob_fail_history']:
    st.markdown("---")
    rc1, rc2, rc3, rc4 = st.columns([3, 1, 1, 1])
    rc1.markdown("**📊 Failure Risk per Hour** · 🟢 <22% safe · 🟡 22–35% watch · 🔴 >35% act now · Blue = smoothed temp")
    pf_now = st.session_state['prob_fail_history'][-1]*100
    rc2.metric("Risk Now", f"{pf_now:.0f}%")
    rc3.metric("Status", "🟢 Safe" if pf_now<22 else "🟡 Watch" if pf_now<35 else "🔴 Act Now")
    if len(st.session_state['prob_fail_history']) >= 2:
        d = (st.session_state['prob_fail_history'][-1] - st.session_state['prob_fail_history'][-2])*100
        rc4.metric("Δ Last Hr", f"{d:+.0f}%", delta_color="inverse")
    fig_rt = risk_chart(st.session_state['prob_fail_history'], st.session_state['ewma_history'])
    st.pyplot(fig_rt, use_container_width=True); plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(f"🍬 Dodol Production Monitor · AI model trained on historical data · proof of concept prototype · 🎯 Target {TEMP_TARGET}°C · 🔴 Upper {UCL:.1f}°C · 🟡 Lower {LCL:.1f}°C")