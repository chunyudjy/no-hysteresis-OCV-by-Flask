# HPPC and OCV Data for Equivalent circuit Modelling

Open Circuit Voltage(OCV) mainly depends on :

- **State of Charge (SoC):** OCV is a direct function of SoC, meaning as SoC increases, OCV also increases in a nonlinear manner. This relationship is determined by the Nernst equation and depends on the electrochemical potential of the electrodes. Each battery chemistry has a characteristic OCV-SoC curve.
OCV 是 SoC 的正函数，这意味着随着 SoC 的增加，OCV 也会以非线性方式增加。该关系由能斯特方程确定，并取决于电极的电化学电位。每种电池化学性质都有其特有的 OCV-SoC 曲线。

- **Temperature:** OCV varies with temperature due to changes in reaction kinetics and electrolyte conductivity. Higher temperatures generally reduce OCV slightly because of increased entropy effects, while lower temperatures increase OCV but may also slow down electrochemical reactions, affecting accuracy in SoC estimation.由于反应动力学和电解液电导率的变化，OCV 会随温度变化。由于熵效应增强，较高的温度通常会略微降低 OCV；而较低的温度会提高 OCV，但也可能减慢电化学反应的速度，从而影响 SoC 估算的准确性。

It also depends on

- **Battery Chemistry**: Different battery chemistries (e.g., Li-ion, NiMH, lead-acid) have distinct OCV characteristics.

- **Aging/State of Health (SoH)**: As a battery degrades, its OCV curve may shift due to changes in internal resistance and capacity loss.

- **Hysteresis Effects**: Some battery chemistries exhibit OCV hysteresis, meaning the voltage depends on whether the battery was previously charged or discharged.

- **Electrode Material Properties**: The composition and structure of electrode materials influence the OCV by affecting ion intercalation/deintercalation.

- **Electrolyte Composition**: The type and concentration of electrolytes impact the chemical potential and, consequently, the OCV.

  

Would you like a more detailed explanation of how each factor influences OCV?

Battery modeling using **Hybrid Pulse Power Characterization (HPPC) and Open Circuit Voltage (OCV) data** involves several key steps:


### **1. Data Collection**

- **HPPC Test:** Perform charge/discharge pulses at different SoC levels to capture dynamic voltage response.

- **OCV Test:** Allow the battery to rest for a long period after charging/discharging to measure equilibrium voltage at different SoC points.

  

### **2. Data Preprocessing**

- Filter out noise from voltage and current signals.

- Align voltage, current, and temperature data with timestamps.

- Extract OCV-SoC data by identifying steady-state voltage after rest periods.

  

### **3. OCV-SoC Curve Fitting**

- Fit the OCV-SoC data to an empirical or polynomial function.

- Use methods like **lookup tables, spline interpolation, or regression models** to represent the OCV-SoC relationship.

  

### **4. Equivalent Circuit Model (ECM) Parameter Identification**

- Use HPPC test data to extract model parameters:

- **Internal Resistance (R0):** From the initial voltage drop after a pulse.

- **Polarization Resistance (R1, R2) and Capacitance (C1, C2):** Derived from voltage relaxation after current pulses.

- Apply **curve fitting or system identification methods** to estimate parameters.

  

### **5. Model Implementation & Validation**

- Implement the identified ECM (e.g., Thevenin or RC model) in **MATLAB, Simulink, or Python**.

- Simulate the battery voltage response under different load conditions.

- Validate the model using test data and compute error metrics (e.g., RMSE, MAE).
为什么温度↑ → OCV↓？
----
```md
**一句话拆解**

|关键词|含义|对 OCV 的影响|
|---|---|---|
|**Higher temperatures**|电池温度升高|① 体系熵（ΔS）↑ ② OCV 略降|
|**Increased entropy effects**|ΔS 与温度项 TΔS 在 G = H − TΔS 中占比更大|当 ΔS < 0（多数锂电反应如此），−ΔS/nF 为正，但 T 增大后 −TΔS/nF 项“吃掉”一部分正电势 → OCV ↓|
|**Lower temperatures**|电池温度降低|① −TΔS/nF 项减小 → OCV 略升 ② 电化学反应速率↓|
|**Slow down electrochemical reactions**|电极/电解液界面的动力学受 Arrhenius 行为支配|内阻 (R = R₀ e^{Eₐ/RT}) ↑，极化电压变大|
|**Affecting accuracy in SoC estimation**|SoC 估算常用 OCV–SoC 曲线|低温下测得的端电压 = OCV + IR_drop，若直接套常温 OCV–SoC 表，会把 IR_drop 当成 SoC 偏差，导致估算失真|

---

### 1 | 为什么温度↑ → OCV↓？

OCV 的热力学表达可由 Nernst 方程的温度微分得到

(∂E∂T)P=−ΔSnF\left(\frac{\partial E}{\partial T}\right)_P = -\frac{\Delta S}{nF}

多数锂离子嵌/脱反应的 ΔS 为 **负值**。

- T↑ → 右式绝对值↑，E 随 T 呈**负斜率** ⇒ OCV 略降。
    
- 典型斜率：−0.15 ~ −0.40 mV K⁻¹，30 °C 升到 50 °C 仅降低几毫伏，但 BMS 精度要求常用 ±2 mV 量级，因此要补偿。
    

---

### 2 | 为什么温度↓ → OCV↑ 但 SoC 难算？

1. **OCV 升高**
    
    - −TΔS/nF 项缩小 ⇒ E 增大，曲线整体上移。
        
2. **动力学变慢**
    
    - 迁移、扩散系数 ~ e^{−Eₐ/RT}，T↓ → 反应阻抗↑
        
    - 放电时端电压 = OCV − I·R_int，低温 R_int 大 → 端电压大幅掉头下滑
        
    - 如果 BMS 仍按常温 OCV–SoC 曲线解算，会把这部分 **IR 极化误判成 SoC 下降** ⇒ 剩余电量被低估。
        

---

### 3 | 工程启示

|温度区间|需关注|常见补偿策略|
|---|---|---|
|**> 40 °C**|OCV 细微下降；副反应加速|温度系数校正 (`dE/dT`)；热失控防护|
|**0 ~ 40 °C**|SoC–OCV 曲线标定点|实时温度补偿曲线拟合|
|**< 0 °C**|端电压大幅极化；充电受限|内阻在线估计 (EKF, OCV‑R 模型) + 低温退流/限流策略|

---

### 4 | 一图速览（概念示意）

```
          OCV
           ↑
           │       ／  (低温)
           │      ／
           │     ／
           │    ／
           │   ／ (常温)
           │  ／
           │ ／
           │/ (高温)
           └───────────→  SoC
```

温度越低，曲线整体“抬高”且更陡；越高则“压低”且更平缓。

---

**总结**

- **热力学层面**：ΔS 是主因，决定 dE/dT 的符号和幅度。
    
- **动力学层面**：R_int 的温度依赖让测得的端电压 ≠ 真正的 OCV。
    
- **BMS 实战**：要同时做 **温度系数补偿 + 内阻线上估计**，否则 SoC 在极端温度下会系统性偏差。
```



# 为什么要用50%SOC的电压差，其他的不可以吗？
🔑为什么首选50 %
50 %正好位于曲线中点，充电段与放电段在这里会“交叉最小化”极化差。把两侧分别线性校正到同一点，可以让低 SOC和高 SOC两端的误差在中间抵消，整体拟合误差最小。

📊数学优势
充/放电电压可近似写成 OCV±k·I·(1‑2 SOC)。在 SOC=0.5 时，±项变零，极化只体现为常量δV。以该点作基准校正，左右两端线性外推即可还原 OCV。

🔋实验可行性
中段电流波动小、温升低，测量噪声最小；极化差 δV 也较稳定，受历史工况影响少，用来标定更可靠。

🛠️能用别的点吗
可以，但需同步修改校正公式：
• 选 x% 时，低段减 x·δV，高段加 (1‑x)·δV
• x 太低或太高会放大另一端误差，且端点容量估计更不稳
• 50 % 综合误差、噪声、实现难度最小，因此最常用。

📈实务建议
若你的电池在 40 %～60 % 区间滞后极小，也可选 40 % 或 60 % 并按上式修正；但结果需与开路静置测得的 OCV 对比验证。


## OCV and SOC
- OCV is equal to terminal voltage when disconnected(open)
- OCV is a function of SOC(State of Charge)
- OCV does not include I*R drop due to current flow nor time dependent effects
- OCV is also a function of temperature
- OCV is key to SOC estimation(ground truth)
- OCV is subject to hysteresis
```
- OCV 等于断开（开路）时的端子电压
- OCV 是 SOC（充电状态）的函数
- OCV 不包括电流引起的 I*R 压降或时间相关效应
- OCV 也是温度的函数
- OCV 是 SOC 估算的关键（基本事实）
- OCV 受迟滞影响
```



# 总体思路
作者要从慢速放电 (step 22 / 23) 和慢速充电 (step 24) 数据中，扣掉 *i·R* 电压降并对齐充放电曲线，使它们在 50 % SOC 处汇合，进而推算“去极化”后的真实 OCV；对 25 °C 算好后，再依次处理其他温度并校正不同温度下的库伦效率 η。

🔢主要步骤
1️⃣**载入指定温度 CSV**→提取放电/充电端电压‑容量‑Step 序列
2️⃣**25 °C**：
 • 计算 η₂₅=∑DisAh/∑ChgAh，用它把充电容量曲线整体缩放到与放电相符
 • 取慢放电段开/终端的电压跳变 IR₂D、慢充电段的跳变 IR₁C/IR₂C，再用经验规则给四个 *i·R* 限值，得到随 SOC 线性过渡的 IRblend
 • 放电端电压+IRblend、充电端电压−IRblend → 近似扣除阻抗后的两条“准 OCV”曲线
 • 取 50 % SOC 处两曲线电压差 ΔV₅₀；低于 0.5 SOC 段用充电线减 z·ΔV₅₀，高于 0.5 用放电线加 (1−z)·ΔV₅₀，使中点汇合
 • 拼接→插值到统一 SOC 栅格 → 得 25 °C 下的 raw OCV
3️⃣**其他温度**：
 • 对每个温度计算自身 ηₖ，同样缩放充电 Ah
 • 用 25 °C 的容量 Q₂₅ 把所有 SOC 归一化到同一坐标
 • 重复 25 °C 的 IRblend、ΔV₅₀ 校正流程 → 得各温度 raw OCV
4️⃣结果保存进 FileData 并逐温度绘图，方便后续拟合成 OCV‑T 模型

⚙️细节注释举例

```python
# ---- 慢放电段（step 22）测尾端 i·R 跌落 ----------
indD  = np.where(p25.s1.disStep == 22)[0]
IR2Da = p25.s1.disV[indD[-1]+1] - p25.s1.disV[indD[-1]]

# ---- 慢充电段（step 24）测首/尾 i·R 上升 ----------
indC  = np.where(p25.s1.chgStep == 24)[0]
IR1Ca = p25.s1.chgV[indC[0]]   - p25.s1.chgV[indC[0]-1]
IR2Ca = p25.s1.chgV[indC[-1]]  - p25.s1.chgV[indC[-1]+1]

# ---- 经验限幅：避免异常噪声导致阻抗过大/过小 ----------
IR1D = 2*IR2Ca                 # 放电起始压降≈2×充电尾端压升
IR2D = min(IR2Da, 2*IR1Ca)     # 放电尾端压降≤2×充电起始压升
IR1C = min(IR1Ca, 2*IR2Da)     # 充电起始压升≤2×放电尾端压降
IR2C = IR2Ca                   # 充电尾端保持原测值
```

🎯这样做的目的
✅去掉瞬时阻抗压降，让曲线接近真正的开路电压
✅用 ΔV₅₀ 统一调平充/放电滞后，让 hysteresis 中点对齐，提高 OCV 精度
✅不同温度下再乘以各自 η，保证容量与 SOC 对应关系一致，从而得出 **OCV = f(SOC, T)** 的原始数据，供后续建模（函数 `OCVfromSOCtemp` 即用于把多温度数据拟合为可调用模型）

## `ir_jumps()` 函数在做什么？

在电池慢速充放电曲线中，\*\*曲线的“首‑尾瞬跳（jump）”\*\*主要来自两部分：

1. **欧姆阻抗（IR drop / IR rise）**

   * 当电流突然由 0 → I（开始充放电）或 I → 0（结束充放电）时，端电压瞬间变化

     $$
       \Delta V_{\text{IR}} = I \times R_{\text{ohm}}
     $$
   * 这一跳跃和真正的化学平衡电压（OCV）无关，必须剔除。

2. **极化/电化学动力学差异**

   * 由于极化，充电曲线整体高于放电曲线；但在极小电流（C/20）下，极化随时间几乎线性衰减，可以用线性内插+ΔV\_{50} 方式统一对齐。
   * 首尾跳跃仍然是最突出的“异常尖点”，因此先行修正它们，后面才能把剩余部分看成“近似平缓曲线”做 ΔV\_{50} 调平。

---

变量含义

| 变量       | 取值点                                   | 物理意义                      | 方向(±) |
| -------- | ------------------------------------- | ------------------------- | ----- |
| **IR1D** | 放电开始瞬间（本文没有直接测，设为 2×IR2C，见下）          | 放电初始 **上升跳**（由于骤停充电后立即放电） | +     |
| **IR2D** | 放电结束瞬间（`dis.iloc[-2] → dis.iloc[-1]`） | 放电尾端 **下降跳**（电流降至 0）      | −     |
| **IR1C** | 充电开始瞬间（`chg.iloc[1] → chg.iloc[0]`）   | 充电初始 **下降跳**（电流由 0→+I）    | −     |
| **IR2C** | 充电结束瞬间（`chg.iloc[-2] → chg.iloc[-1]`） | 充电尾端 **上升跳**（+I→0）        | +     |

> 注意：符号取决于“升/降”，代码中直接用电压差 `V[n‑1] − V[n]`，无需显式写正负号。

---

计算逻辑逐行拆解

```python
IR2D = dis["Voltage"].iloc[-2] - dis["Voltage"].iloc[-1]
```

* **放电尾端**的两点差值 `V₋₂ − V₋₁`。
* 放电在进行中时电流为 −I，最终停到 0 A；因此 **电压会抬高**（减小 IR drop），`V[-1]` > `V[-2]`，差值**为负**，代表一次 **向上跳跃** 的幅值。

```python
IR1C = chg["Voltage"].iloc[1]  - chg["Voltage"].iloc[0]
```

* **充电开始**瞬间，电流 0 → +I，端电压会 **立刻下降**；所以差值通常 **为正**（`V0` < `V1`）。

```python
IR2C = chg["Voltage"].iloc[-1] - chg["Voltage"].iloc[-2]
```

* **充电结束**时 +I → 0，端电压抬高；差值通常 **为正**。

---

经验限幅（两行 `min()` 和倍数）

真实数据可能含噪声或操作不完全对称：

* 若直接使用 4 个差值，可能出现“某个跳跃异常大/小”，导致后续线性内插出现畸变。
* 常见经验规则：**同一段充/放首尾跳跃幅值应同量级**。因此采用：

```python
IR1D = 2 * IR2C        # 若缺实测点，用充电尾跳 ×2 估算放电首跳
IR2D = min(IR2D, 2*IR1C)
IR1C = min(IR1C, 2*IR2D)
```

1. **放电开始（IR1D）** 直接缺少观测（因为真正放电首点在前一充电尾之后，可能被跳过）。

   * 取 **充电尾跳**（IR2C）×2 作为粗略上限：放电首跳一般更大，因为从 +I 立刻切到 −I。
2. **IR2D** 与 **IR1C** 互相设限，避免一个极端值造成线性校正“过补/欠补”。

---

最终返回四个值干嘛用？

```python
return IR1D, IR2D, IR1C, IR2C
```

* `calc_ocv()` 中将它们插进 **线性混合**：

```python
blendD = np.linspace(0,1,len(dis))          # 0→1 放电序列
disVcor = dis.V + (IR1D + (IR2D-IR1D)*blendD)

blendC = np.linspace(0,1,len(chg))          # 0→1 充电序列
chgVcor = chg.V - (IR1C + (IR2C-IR1C)*blendC)
```

⬇️ **目的**

1. **起点**：给整段曲线加/减各自的首跳（IR1D, IR1C）。
2. **终点**：线性过渡到尾跳（IR2D, IR2C）。
3. 结果是把 *两条曲线* 的“突兀阶梯”削成几乎连续的斜率，后续才能用 **ΔV₅₀** 把充放电两段再无缝拼接。

---

可视化示意

```
原始 (示意)                    校正后 (示意)
  V                              V
  │   ▄▄▄▄←IR1D                  │
4.2│  █                          │
    █  ▄▄▄▄←IR2D                 │
    █ █                          │
4.0│█ █    (discharge)           │█ (dis)
    █ █                          │ █
    █ █                          │ █
3.8│█ █                          │ █
    █ █                          │ █
    █▄█                          │ █▄▄
    ▲  ▲  ▲  t                  ▲  ▲  ▲  t
    |  |  |                      |  |  |
   IR1C IR2C                    已线性补偿
```

---

小结

* `ir_jumps()` 利用 **首末两个采样点** 简单估算四个瞬时 IR 跳跃幅值。
* 再通过极简经验限幅避免噪声。
* 这些跳跃随后用 **线性融合** 的方式从原始电压中扣除，从而获得“移除欧姆阻抗 + 极化趋势一阶近似”的电压曲线，为 ΔV₅₀ 调平和最终 OCV 插值打下基础。


# VersionUp
もし今後さらに機能追加（多ファイル一括処理・パラメータ調整 UI・Plotly への切り替えなど）やデプロイ方法（Docker 化、社内サーバへの設置など）が必要になったら、いつでも気軽に声をかけてくださいね。


