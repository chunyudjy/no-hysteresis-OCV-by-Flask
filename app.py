#!/usr/bin/env python3
# app.py
import io, base64, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for
)

app = Flask(__name__)

# ────────────────── 可按需要修改 ──────────────────
STEPS_DIS = (22, 23)
STEPS_CHG = (23, 24, 25)
SKIPROWS  = 28
SOC_GRID  = np.arange(0, 1.005, 0.005)
# ────────────────────────────────────────────────

# ---------- 数据处理核心函数（与之前相同，略有注释精简） ----------
def read_cell_csv(file_storage):
    df = (pd.read_csv(file_storage, skiprows=SKIPROWS)
            .drop(columns=lambda c: c.startswith("Unnamed"), errors="ignore"))
    df.columns = ["Time","Step","Status","Prog Time","StepTime","Cycle",
                  "CycleLv","Procedure","Voltage","Current","Temp",
                  "Capacity","Wh","Cnt","Empty"]
    df["Prog Time"] = pd.to_timedelta(df["Prog Time"])
    for c in ["Voltage","Current","Capacity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("Prog Time").reset_index(drop=True)

def extract_sections(df):
    dis = df[df["Step"].isin(STEPS_DIS)].copy()
    chg = df[df["Step"].isin(STEPS_CHG)].copy()
    Qmax = -dis["Capacity"].min()
    dis["SoC"] = (Qmax + dis["Capacity"]) / Qmax
    chg["SoC"] = (chg["Capacity"] - chg["Capacity"].min()) / \
                 (chg["Capacity"].max() - chg["Capacity"].min())
    return dis.reset_index(drop=True), chg.reset_index(drop=True)

def ir_jumps(dis, chg):
    IR2D = dis["Voltage"].iloc[-2] - dis["Voltage"].iloc[-1]
    IR1C = chg["Voltage"].iloc[1]  - chg["Voltage"].iloc[0]
    IR2C = chg["Voltage"].iloc[-1] - chg["Voltage"].iloc[-2]
    IR1D = 2 * IR2C
    IR2D = min(IR2D, 2 * IR1C)
    IR1C = min(IR1C, 2 * IR2D)
    return IR1D, IR2D, IR1C, IR2C

def calc_ocv(dis, chg):
    IR1D, IR2D, IR1C, IR2C = ir_jumps(dis, chg)
    blendD   = np.linspace(0, 1, len(dis))
    blendC   = np.linspace(0, 1, len(chg))

    disVcor  = dis["Voltage"] + (IR1D + (IR2D - IR1D) * blendD)
    chgVcor  = chg["Voltage"] - (IR1C + (IR2C - IR1C) * blendC)

    dV50 = np.interp(0.5, chg["SoC"], chgVcor) \
         - np.interp(0.5, dis["SoC"][::-1], disVcor[::-1])

    zC = chg["SoC"][chg["SoC"] < 0.5]
    vC = chgVcor[chg["SoC"] < 0.5] - zC * dV50
    zD = dis["SoC"][dis["SoC"] > 0.5]
    vD = disVcor[dis["SoC"] > 0.5] + (1 - zD) * dV50

    soc_sorted = np.concatenate([zC, zD[::-1]])
    v_sorted   = np.concatenate([vC, vD[::-1]])
    ocv = np.interp(SOC_GRID, soc_sorted, v_sorted)
    return SOC_GRID, ocv, dis["SoC"], dis["Voltage"], chg["SoC"], chg["Voltage"]

def plot_b64(soc, ocv, disZ, disV, chgZ, chgV):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(100*soc, ocv, label='Approximate OCV from Data', lw=2, c='b')
    ax.plot(100*disZ, disV, '--', c='k', lw=1.1, label='Raw Discharge')
    ax.plot(100*chgZ, chgV, '--', c='r', lw=1.1, label='Raw Charge')
    ax.set_xlabel('State of Charge (%)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Open Circuit Voltage (OCV) vs SOC without hysteresis')
    ax.legend(); ax.grid(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------ 页面路由 ------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=False, skip=SKIPROWS)

@app.route("/", methods=["POST"])
def process():
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for('index'))

    f = request.files["file"]
    df         = read_cell_csv(f)
    dis, chg   = extract_sections(df)
    soc, ocv, disZ, disV, chgZ, chgV = calc_ocv(dis, chg)

    # 图 → base64
    img_b64 = plot_b64(soc, ocv, disZ, disV, chgZ, chgV)

    # CSV → base64 data URI
    csv_name = Path(f.filename).stem + "_OCV-without-hysteresis.csv"
    csv_buf  = io.StringIO()
    pd.DataFrame({"SOC": soc, "Approximate OCV from Data": ocv})\
        .to_csv(csv_buf, index=False)
    csv_data_uri = "data:text/csv;base64," + \
                   base64.b64encode(csv_buf.getvalue().encode()).decode()

    # 把结果发送回前端
    return render_template(
        "index.html",
        result=True,
        plot_b64=img_b64,
        csv_name=csv_name,
        csv_data=csv_data_uri,
        skip=SKIPROWS
    )

# ------------ 纯 API（可选） -------------
@app.route("/api/ocv", methods=["POST"])
def api_ocv():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    df = read_cell_csv(request.files["file"])
    dis, chg = extract_sections(df)
    soc, ocv, *_ = calc_ocv(dis, chg)
    return jsonify({"soc": soc.tolist(), "ocv": ocv.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=8051)
