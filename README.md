# Industrial Battery Energy Storage System (BESS) EIS Analytics Pipeline

An industrial-grade telemetry and diagnostics data pipeline designed to analyze Electrochemical Impedance Spectroscopy (EIS) measurements for grid-scale Lithium Iron Phosphate (LFP) battery storage systems. 

## 🚀 Key Features
- **Built-in Synthetic EIS Simulation Engine:** Generates mathematically rigorous EIS spectra with custom measurement noise, eliminating hardware file dependencies and making the repository instantly executable out-of-the-box.
- **Automated Milliohm Scaling:** Tailored specifically for utility-scale applications (e.g., 280Ah commercial cells), mapping all complex vectors directly into the milliohm ($m\Omega$) domain.
- **Robust Circuit Fitting:** Utilizes a non-linear Least-Squares solver (Trust Region Reflective algorithm with a robust Soft-L1 loss function) to fit data to an advanced **Dual-Arc Battery Model** ($Rs + L + [Rsei \parallel CPEsei] + [Rct \parallel CPEdl] + W$).
- **SCADA/EMS Telemetry Payloads:** Extracts the high-frequency purely Ohmic intercept ($R_0$), computes a standardized State of Health (SOH) indicator, and compiles production-ready JSON and Excel reporting architectures.
- **Publication-Quality Graphics:** Renders clean, high-DPI Nyquist, dual-axis Bode, and spatial model fitness residual plots with optimized external legend rendering.

## 📁 Repository Architecture
- `eis_battery_diagnostics.py`: Core python processing automation layer.
- `Untitled.ipynb`: Interactive Jupyter Notebook executing the full operational loop.
- `requirements.txt`: Python package dependency map.
- `screenshots/`: Automated visualization output exports.

## 📊 Analytics Documentation Samples
The pipeline automatically outputs the following standard validation graphics:
1. **Nyquist Plot:** Isolating Ohmic resistance and profiling the high-frequency inductive tail.
2. **Bode Plot:** Parallel spectrum tracking of impedance modulus and phase angle.
3. **Fit Residuals:** Error-vector alignment verification to eliminate underfitting.
