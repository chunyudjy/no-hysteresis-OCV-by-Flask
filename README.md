# OCV‑No‑Hysteresis WebApp

⚡ A lightweight Flask application that extracts **open‑circuit voltage (OCV) curves without hysteresis** from single `_C20DisCh.csv` files and provides:

* 📈 Instant visualization of the corrected OCV vs SOC curve  
* ⬇️ One‑click download of the generated SOC–OCV CSV  
* 🌐 A minimal REST API (`/api/ocv`) for programmatic access

> Ideal for lithium‑ion battery researchers who need a quick, browser‑based tool to inspect and export hysteresis‑free OCV data.
<img width="739" height="575" alt="Open Circuit Voltage (0CV) vs SOC without hysteresis" src="https://github.com/user-attachments/assets/a4d0d70a-1902-4dab-873b-47658e9dfb31" />

---

## Features

| Feature | Description |
| ------- | ----------- |
| **Drag‑and‑drop upload** | Upload any `*_C20DisCh.csv`; skip‑rows and step codes are configurable in `app.py`. |
| **IR‑drop + ΔV<sub>50</sub> correction** | Replicates the core algorithm from notebook scripts—fully vectorized with NumPy & Pandas. |
| **High‑quality PNG plot** | Matplotlib backend (`Agg`) renders the “OCV vs SOC without hysteresis” curve alongside raw charge / discharge data. |
| **CSV export** | Generates `original_name_OCV-without-hysteresis.csv` containing two columns: `SOC` and `Approximate OCV from Data`. |
| **JSON API** | `POST /api/ocv` with a multipart CSV → returns `{soc: [...], ocv: [...]}`. |

---

## Quick Start

### 1 · Clone

```bash
git clone https://github.com/<your-username>/ocv-no-hysteresis-webapp.git
cd ocv-no-hysteresis-webapp
