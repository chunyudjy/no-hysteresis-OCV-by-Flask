## 1.print("All matching files:", files) 如果有多个files怎么让他们垂直排列?

有三种常用写法让列表中的文件逐行输出：

```python
# 1. 最直观的写法：先打印标题，再逐行打印
print("All matching files:")
for f in files:
    print(f)

# 2. 一条语句完成
print("All matching files:\n" + "\n".join(files))

# 3. 利用 print 的 * 解包与 sep
print("All matching files:", *files, sep="\n")
```

三种效果一致：
All matching files:
file1.csv
file2.csv
file3.csv

## 2.* 解包是什么作用?

在 Python 函数调用中，前面加 \* 的对象会被“解包”成一个个独立的位置参数。

```python
def func(a, b, c):
    print(a, b, c)

lst = [1, 2, 3]
func(*lst)          # 等价于 func(1, 2, 3)
```

在 `print(*files, sep="\n")` 里，`*files` 把列表 `files` 中的每个元素当成单独的实参传给 `print`；再配合 `sep="\n"` 指定换行分隔，所以能够逐行输出。

常见用法

1. **作为实参解包**

   * `sum(*[1,2,3])` 等价于 `sum(1,2,3)`
   * `open(*["data.txt", "r"])` 等价于 `open("data.txt", "r")`
2. **序列解包赋值**（星号可以在左侧）

   ```python
   a, *rest = [10, 20, 30, 40]   # a=10, rest=[20,30,40]
   *head, tail = [10, 20, 30]    # head=[10,20], tail=30
   ```
3. **在字典解包时用 `**`**

   * `func(**{"a":1,"b":2,"c":3})` 把键值对解包为关键字参数
   * 合并字典：`new = {**d1, **d2}`

核心记忆点：`*` 负责把**可迭代对象拆成位置参数**，`**` 则把**映射类型拆成关键字参数**。


## TypeError: 'value' must be an instance of str or bytes, not a float
```

<class 'pandas.core.frame.DataFrame'> RangeIndex: 2422 entries, 0 to 2421 Data columns (total 15 columns): # Column Non-Null Count Dtype --- ------ -------------- ----- 0 Time Stamp 2422 non-null object 1 Step 2422 non-null int64 2 Status 2422 non-null object 3 Prog Time 2422 non-null object 4 Step Time 2422 non-null object 5 Cycle 2422 non-null int64 6 Cycle Level 2422 non-null int64 7 Procedure 2422 non-null object 8 Voltage 2422 non-null float64 9 Current 2422 non-null float64 10 Temperature 2422 non-null float64 11 Capacity 2422 non-null float64 12 WhAccu 2422 non-null float64 13 Cnt 2422 non-null float64 14 Empty 0 non-null float64 dtypes: float64(7), int64(3), object(5) memory usage: 284.0+ KB

plt.plot(df_OCV['Prog Time'], df_OCV.Voltage)

TypeError: 'value' must be an instance of str or bytes, not a float
```

`Prog Time` 这一列现在是 `object` 类型，里头混杂了字符串（如 "00:05:30"）和缺失值 `NaN`（以 `float` 形式存在）。
当 `plt.plot` 发现 x 轴数据是对象类型时，会尝试把它们解析成日期/时间；一旦碰到 `float`，就抛出

```
TypeError: 'value' must be an instance of str or bytes, not a float
```

 解决思路

1. **先把 `Prog Time` 统一转换成真正的时间 / 时间差类型**（`Timedelta` 或 `datetime`）
2. **把无法解析的值设为 `NaT` 并丢掉**
3. **再绘图**

下面给出两种常见做法。任选其一即可。

---

 做法 A：把 `Prog Time` 当作“累积运行时间”来画

如果这一列长得像 `HH:MM:SS`（或 `D:HH:MM:SS`），代表从开始计时到当前点的**累计时长**，可用 `Timedelta`：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1) 解析为 Timedelta，无法解析的设为 NaT
df_OCV['Prog Time'] = pd.to_timedelta(df_OCV['Prog Time'], errors='coerce')

# 2) 去掉 NaT 行
df_plot = df_OCV.dropna(subset=['Prog Time'])

# 3) 用秒数或分钟数当横轴
x_seconds = df_plot['Prog Time'].dt.total_seconds()

plt.figure(figsize=(6,4))
plt.plot(x_seconds, df_plot['Voltage'])
plt.xlabel("Prog Time (s)")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.show()
```

---

做法 B：把 `Prog Time` 当作具体的日期时间来画

如果它看起来像 `"2025-07-26 14:35:00"` 之类的时间戳，用 `to_datetime` 处理即可：

```python
df_OCV['Prog Time'] = pd.to_datetime(df_OCV['Prog Time'], errors='coerce')

df_plot = df_OCV.dropna(subset=['Prog Time'])

plt.figure(figsize=(6,4))
plt.plot(df_plot['Prog Time'], df_plot['Voltage'])
plt.xlabel("Time")
plt.ylabel("Voltage (V)")
plt.gcf().autofmt_xdate()   # 让日期标签自动斜排
plt.tight_layout()
plt.show()
```

---

### 小结

* **核心问题**：`Prog Time` 列里混进了 `NaN`，导致类型不统一。
* **关键修复**：`pd.to_timedelta` 或 `pd.to_datetime` + `errors='coerce'`，把非法值变 `NaT`，再 `dropna`。
* 之后再 `plt.plot` 就不会出现 “必须是 str 或 bytes” 的报错了。


## Prog Time 长这样：29:20:19.688

`Prog Time` 的格式是 **HH\:MM\:SS.sss**，其中小时数可超过 24，属于“累计时间”。直接用 `pd.to_timedelta` 就能解析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. 解析为 Timedelta；无法识别的值会变成 NaT
df_OCV['Prog Time'] = pd.to_timedelta(df_OCV['Prog Time'], errors='coerce')

# 2. 去掉 NaT 行
df_plot = df_OCV.dropna(subset=['Prog Time'])

# 3. 取累计秒数或分钟数作横轴
x_sec = df_plot['Prog Time'].dt.total_seconds()

plt.figure(figsize=(6,4))
plt.plot(x_sec, df_plot['Voltage'])
plt.xlabel("Prog Time (s)")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.show()
```

要用分钟：`x_min = df_plot['Prog Time'].dt.total_seconds() / 60`。
如果只想看相对时间轴（0 开始），再减去 `x_sec.iloc[0]` 即可。

## np.arange(len(df_dis))?

`np.arange(len(df_dis))` 只是用 **NumPy** 生成一个等差整数序列：

| 步骤             | 说明                                    |
| -------------- | ------------------------------------- |
| `len(df_dis)`  | 返回 `df_dis` 这一 DataFrame 的行数，例如 2422。 |
| `np.arange(n)` | 产生从 0 到 n‑1 的整型数组（间隔 1）。              |

```python
>>> len(df_dis)
2422
>>> np.arange(len(df_dis))
array([   0,    1,    2, ..., 2419, 2420, 2421])
```

### 常见用途

1. **索引用作 x 轴**
   当真正的时间戳不好用、或你只关心相对顺序时：

   ```python
   plt.plot(np.arange(len(df_dis)), df_dis['Voltage'])
   ```
2. **创建新的整型索引**

   ```python
   df_dis.index = np.arange(len(df_dis))
   ```
3. **向量化运算或分组时需要序号**
   例如按奇偶行拆分：

   ```python
   even_rows = df_dis[np.arange(len(df_dis)) % 2 == 0]
   ```

如果你想让序列从 1 开始，用 `np.arange(1, len(df_dis)+1)`，或者直接使用 Pandas 的 `df_dis.reset_index(drop=False)` 增加一列原始索引。


## plt调整x轴的tick?

常用的 x 轴刻度(tick) 调整方法：

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
```

1. **直接指定刻度位置与标签**

```python
ticks = np.arange(0, 11, 2)                # 0,2,4,6,8,10
labels = [f"{t} s" for t in ticks]          # 自定义文字
ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=45)    # rotation 旋转角度
```

2. **限制刻度数量**（`MaxNLocator` 自动均分）

```python
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))  # 最多 5 个主刻度，首尾可裁剪
```

3. **按固定间隔放置刻度**（`MultipleLocator`）

```python
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(2))   # 每 2 单位一个主刻度
ax.xaxis.set_minor_locator(MultipleLocator(0.5)) # 每 0.5 单位一个次刻度
```

4. **日期/时间轴刻度**

```python
import pandas as pd
import matplotlib.dates as mdates
dates = pd.date_range("2025-07-26 12:00", periods=100, freq="T")
y = np.sin(np.linspace(0, 10, len(dates)))
fig, ax = plt.subplots()
ax.plot(dates, y)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))        # 每小时一个主刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))       # 格式 HH:MM
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))     # 每 15 min 次刻度
fig.autofmt_xdate()                                               # 标签自动旋转
```

5. **隐藏刻度或标签**

```python
ax.tick_params(axis='x', which='both', length=0)   # 不显示刻度线
ax.set_xticklabels([])                             # 不显示刻度文字
```

6. **移动刻度到顶部**

```python
ax.xaxis.tick_top()
```

组合运用即可满足大多数需求。

## y轴反转问题

```
目的：绘制 相对时间vs电压
电压先从满电态4.2V降低至3.0V然后充电至4.2V，所以蓝线应该是随着相对时间下降，红色是上升，但是这个代码做的图两个都是上升，而且y轴是3.0V在顶部，而且应该都不是直线才对。

代码：
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots()                                   # ← 拿到 ax 对象
ax.plot(np.arange(len(df_dis)), df_dis['Voltage'],
        label="Discharge", color='b')

ax.plot(np.arange(len(df_chg)), df_chg['Voltage'],
         label="Charge", color='r')

ax.set_xlabel("Index")
ax.set_ylabel("Voltage (V)")
ax.set_title("Voltage Profile")
ax.legend()
ax.grid()

# 关键：y 轴主刻度最多 5 个
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

plt.show()

```

造成 “蓝线红线都朝上、3 V 出现在顶部” 的根本原因，是 **x 轴并不是你想要的“相对时间”**，而 `Voltage` 列顺序也被打乱／反转了。下面给出一套“从原始表格 → 正确曲线” 的完整示例，你可以直接粘贴运行，对照排查自己的数据。

---

 1 读入并解析时间列

```python
import pandas as pd

df = pd.read_csv("your_file.tsv", sep="\t")      # ← 逗号分隔改成 sep="," 即可

# 把字符串变成真正的时间
df["Time Stamp"] = pd.to_datetime(
    df["Time Stamp"],            # 10/26/2018 9:29:22 AM
    format="%m/%d/%Y %I:%M:%S %p"
)

# 把 HH:MM:SS.sss（小时可>24）变成 Timedelta
df["Prog Time"] = pd.to_timedelta(df["Prog Time"])

# Voltage 一定要是 float
df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
```

---

 2 按充/放电拆分并**保证时间顺序**

有些设备会把放电段写在文件中前面，有的则相反。最稳妥办法是：

* 用 `Prog Time` 或 `Time Stamp` 排序
* 然后再按 `Status` 拆分

```python
df = df.sort_values("Prog Time").reset_index(drop=True)

df_dis = df[df["Status"] == "DCH"].copy()
df_chg = df[df["Status"] == "CHG"].copy()        # 如果有充电段
```

---

 3 创建 “相对时间（秒）” 轴

以整条曲线的 **起点** 为 0 s，便于首尾拼接：

```python
t0 = df["Prog Time"].iloc[0]
df["t_rel"] = (df["Prog Time"] - t0).dt.total_seconds()

t_dis = df_dis["t_rel"]
t_chg = df_chg["t_rel"]
```

---

 4 绘图 & y 轴方向检查

```python
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(t_dis, df_dis["Voltage"], label="Discharge", color="b")
if not df_chg.empty:
    ax.plot(t_chg, df_chg["Voltage"], label="Charge", color="r")

ax.set_xlabel("Relative Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title("Voltage vs. Relative Time")
ax.legend()
ax.grid()

# y 轴主刻度最多 5 个
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# === 检查 y 轴是否倒了 ===
ymin, ymax = ax.get_ylim()
if ymin > ymax:          # 倒挂就翻过来
    ax.invert_yaxis()

plt.tight_layout()
plt.show()
```

 结果符合预期应为

* **蓝线（放电）**：从 4.2 V **向下** 到约 3.0 V
* **红线（充电）**：再 **向上** 回到 4.2 V
* y 轴上端是 4.3 V 左右，下端是 3 V 左右

---

 5 如果仍然方向不对？

1. **电压列是否被反转？**

   ```python
   print(df_dis["Voltage"].head(), df_dis["Voltage"].tail())
   ```

   首部应高于尾部。若相反，说明设备把最后的数据写在文件开头，需要 `sort_values(...)`。
2. **数字读成字符串**：
   `df["Voltage"].dtype` 应该是 `float64`。否则先 `pd.to_numeric`。
3. **y 轴倒挂**：确认代码没有其它地方调用 `ax.invert_yaxis()` 或 `set_ylim(高, 低)`。

跟着上面的流程走完，一般都能修正 “两条线都朝上 / 3 V 在顶部 / 直线” 等异常现象。


## [::-1]如何做到逆序绘制?

`[::-1]` 是 **Python 切片语法** `[开始:结束:步长]` 中最简短的一种“反向切片”写法。

* 省略 `开始` → 从序列**最后一个元素**开始
* 省略 `结束` → 一直取到序列最前端
* `步长 = -1`  → 每次向左移动 1 个位置

于是整个序列就被倒着复制出来了。

```python
lst = [10, 20, 30, 40]
print(lst[::-1])           # [40, 30, 20, 10]
```

 通用记忆模板

```
seq[start : stop : step]
```

| 位置      | 允许省略 | 默认值                             | 说明       |
| ------- | ---- | ------------------------------- | -------- |
| `start` | ✓    | 0（正向）<br>len(seq)-1（反向）         | 起点索引     |
| `stop`  | ✓    | len(seq)（正向）<br>-len(seq)-1（反向） | 终点 *前一位* |
| `step`  | ✗    | 1                               | 步长，可正可负  |

当 `step` 为负数时，解释器自动把省略的 `start`、`stop` 调整为“**从末尾走到开头**”的合理缺省值，所以 `[::-1]` 就是“整个序列倒序遍历”的快捷方式。

---

 用在不同对象上的效果

| 对象                | 代码示例                       | 结果                  |
| ----------------- | -------------------------- | ------------------- |
| **list**          | `[1, 2, 3][::-1]`          | `[3, 2, 1]`         |
| **NumPy 数组**      | `arr[::-1]`                | 视图 (view) 倒序，不占额外内存 |
| **Pandas Series** | `s[::-1]` 或 `s.iloc[::-1]` | 索引顺序也随之反转           |
| **字符串**           | `"abcd"[::-1]`             | `"dcba"`            |

> **注意**
>
> * 对 Pandas 来说，如果索引是数字且不连续，推荐写 `s.iloc[::-1]` 更明确。
> * 切片会返回一个**拷贝**（list、Series）或**视图**（NumPy ndarray）。对结果做修改不会影响原对象，除非是 NumPy 视图并且你直接写值。

---

 应用在放电曲线

```python
# 反转电压列，使 x=0 对应满电，x=max 对应欠电
ax.plot(np.arange(len(df_dis)), df_dis['Voltage'][::-1],
        label="Discharge", color='b')
```

如果原始数据文件把放电段“欠电→满电”写在文件顶部，为了让曲线方向与真实放电过程一致，就先用 `[::-1]` 把它颠倒过来再绘图。

这样就能直观理解 `[::-1]` 的作用：**让任何可切片序列“一键倒序”**。


## Interpolated Voltage Profile Between Charge & Discharge"

下面逐行拆解这段脚本在做什么、为什么这样做，以及每一步的关键细节。

| 代码片段                                                                                                            | 作用                                                                                                                                                                                                                                                                                 | 细节 & 关键点                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `python<br>import numpy as np<br>import matplotlib.pyplot as plt<br>from scipy.interpolate import interp1d`     | 导入所需库                                                                                                                                                                                                                                                                              | *NumPy* 处理数组，*Matplotlib* 画图，*SciPy* 的 `interp1d` 构建 1 维插值模型。                                                               |
| `python<br># Ensure both series have the same length …<br>min_len = min(len(df_dis), len(df_chg))`              | 两条曲线往往采样点数不同。为便于对齐，将两条曲线都**裁剪**到最短长度 `min_len`。                                                                                                                                                                                                                                    | 这样做可以避免插值时出现「数组长度不一致」的错误。                                                                                                   |
| `python<br>x_common = np.linspace(0, min_len - 1, min_len)`                                                     | 构造一个公共的 x 轴 0 — (min\_len − 1)。                                                                                                                                                                                                                                                    | 用于给所有曲线（放电、充电、插值曲线）统一横坐标。                                                                                                   |
| `python<br># Interpolation model …<br>interp_model = interp1d([0, 1], np.vstack([...]), axis=0, kind='linear')` | \*\*核心步骤：\*\*用 `interp1d` 在“放电 → 充电”之间做线性插值。<br>- `np.vstack([...])` 把<br> • 放电电压 (反转) **→** 第 0 行<br> • 充电电压  **→** 第 1 行<br>  叠成形状 (2, min\_len) 的 2 × N 数组。<br>- `[0, 1]` 是“插值参数”方向的自变量。<br>- `axis=0` 指明 **沿第 0 维**（两条曲线之间）做插值。<br>- `kind='linear'` 线性插值，可改 `'cubic'` 得到平滑曲线。 | 结果得到的 `interp_model(param)` 接收 **0 – 1** 之间的参数：<br>- `param=0` ⇒ 100 % 放电曲线<br>- `param=1` ⇒ 100 % 充电曲线<br>- 介于其间 ⇒ 两者的加权平均 |
| `python<br># Generate interpolated values in between<br>interp_steps = np.linspace(0, 1, 10)`                   | 生成 10 个等间隔参数（含 0 和 1）                                                                                                                                                                                                                                                              | 用来画 8 条中间过渡曲线 + 2 条端点曲线（实际端点后面又单独画了）。                                                                                       |
| `python<br>for step in interp_steps:<br>    plt.plot(x_common, interp_model(step), … color='g', alpha=0.2)`     | 循环画出 10 条“渐变”曲线                                                                                                                                                                                                                                                                    | `alpha=0.2` 让过渡曲线半透明、淡绿色，看起来像一片渐变带。                                                                                         |
| `python<br># Plot Discharge, Charge, and Interpolation`                                                         | 再分别把真·放电、真·充电曲线画出来                                                                                                                                                                                                                                                                 | - 放电用 `[::-1]` 反转，使其 **从满电 4.2 V → 欠电 3 V** 的方向与充电首尾衔接。<br>- 颜色分别蓝、红。                                                       |
| `python<br>plt.xlabel("Index") … plt.show()`                                                                    | 常规收尾：坐标轴标签、标题、网格、图例                                                                                                                                                                                                                                                                | `Index` 只是点号；若用秒/分钟，应换成前面计算的 `t_rel`。                                                                                       |

 直观效果

* **蓝线**：放电曲线（从高到低）。
* **红线**：充电曲线（从低到高）。
* **一束浅绿过渡带**：放电→充电之间的线性插值，越靠近蓝线越像放电，越靠近红线越像充电。
* 形成类似“电压—荷电状态回环”的填充视觉，可用来形象展示充放电 **滞后 / Hysteresis**。

 常见扩展

1. **改用 `kind='cubic'`**

   ```python
   interp_model = interp1d([0, 1], …, kind='cubic')
   ```

   曲线更平滑，但两端会保持一致。
2. **使用真实时间轴**
   把 `x_common` 换成 `t_rel` 或 `Prog Time`，更符合物理意义。
3. **可视化差异**
   还可计算放电与充电电压的差值 `df_dis[::-1] - df_chg`，评估 IR Drop / 极化。

这样整段脚本的**目的**就很清晰：

> **把放电与充电曲线首尾对齐 → 在两条曲线之间插值 → 画出一组“渐变”曲线，直观展示二者的过渡与差异。**


## errors="coerce"是什么意思？

在 pandas 的各种“类型转换”函数里（`pd.to_numeric`、`pd.to_datetime`、`pd.to_timedelta` 等），参数 **`errors`** 用来指定**遇到无法解析的值时该怎么办**：

| 取值               | 行为                                                                  | 典型结果                               |
| ---------------- | ------------------------------------------------------------------- | ---------------------------------- |
| `'raise'` *(默认)* | 只要有一个值解析失败，就立刻抛出 `ValueError`，整个转换终止                                | 程序报错                               |
| `'coerce'`       | \*\*“强制”转换：\*\*能解析的正常转换，解析失败的统统变成空值<br>- 数值 → `NaN`<br>- 时间 → `NaT` | 转换成功但出现缺失值，可后续 `dropna` / `fillna` |
| `'ignore'`       | 什么都不做，直接返回原对象（保持原始 `dtype`）                                         | 数据不变，常用得很少                         |

 简易示例

```python
import pandas as pd

s = pd.Series(['12.3', 'bad', '5'])

# 1) 默认 errors='raise' → 抛错
# pd.to_numeric(s)
# ValueError: Unable to parse string "bad" at position 1

# 2) errors='coerce' → 把 "bad" 变 NaN
clean = pd.to_numeric(s, errors='coerce')
print(clean)
# 0    12.3
# 1     NaN   ← 解析失败 → NaN
# 2     5.0
# dtype: float64
```

在数据清洗场景中，`errors='coerce'` 十分常用——

* 先把“脏”字符、安全地转成 `NaN`/`NaT`，
* 再 `dropna()` 或 `fillna()` 统一处理，
  就能避免因单个异常值导致整列转换失败。
