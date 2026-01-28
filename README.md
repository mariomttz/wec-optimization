<h1 align="center">
  Optimization of Wave Energy Farms Using Surrogate Models and Metaheuristics
</h1>

<div align="center">
  
  ![Python Version](https://img.shields.io/badge/python-3.10.12-blue.svg)
  ![Frameworks](https://img.shields.io/badge/frameworks-PyTorch%20%7C%20Scikit--learn%20%7C%20CatBoost%20%7C%20Optuna%20%7C%20fastai%20%7C%20Pandas%20%7C%20Numpy-orange.svg)

</div>

This repository contains the official source code, datasets, and results for the research paper **"Optimization of Wave Energy Farms Using Surrogate Models and Metaheuristics"**. Our work focuses on optimizing the spatial placement of Wave Energy Converters (WECs) to maximize the power output of a wave farm. 

> [!NOTE]
> This work was carried out with the support of the **UNAM-DGAPA-PAPIME PE101325** Program.

---

## 👥 Authors

* **Mario Alberto Martinez Oliveros** (Graduate)
* **Dra. Adriana Menchaca Méndez** (Advisor)
* **Dr. Miguel Raggi Pérez** (Advisor)

**Affiliation:**
*Escuela Nacional de Estudios Superiores (ENES), Unidad Morelia, Universidad Nacional Autónoma de México (UNAM), México.*

---

## 📖 About the Project

Designing wave energy farms is a complex task, primarily due to the high computational cost of hydrodynamic simulators used to evaluate the farm's power output. This research addresses this challenge by employing machine learning models as **surrogate objective functions**, which drastically reduces computation time.

We conducted a comparative study of four regression models:
* Linear Regression 
* Support Vector Machines (SVM) 
* CatBoost 
* Deep Neural Networks (DNN)

The best-performing model was then used to find the optimal buoy placement using two well-known metaheuristics:
* **Differential Evolution (DE)**
* **Particle Swarm Optimization (PSO)**

Our findings show that combining **Support Vector Machines** as a surrogate model with **Differential Evolution** provides a highly effective and efficient alternative for designing wave energy farms, achieving superior results compared to those obtained using direct hydrodynamic simulation.

---

## 📂 Repository Structure

The repository is organized as follows:

```
.
├── data/                  # Contains raw and cleaned datasets.
│   ├── raw/               # Original data from the UCI repository.
│   └── clean/             # Processed data used for model training.
├── etc/                   # Utility scripts for analysis and file management.
├── models/                # ML models, training scripts, and results.
│   ├── perth_49/
│   ├── perth_100/
│   ├── sydney_49/
│   └── sydney_100/
├── optimization/          # Metaheuristic optimization scripts and results.
│   ├── perth_49/
│   ├── perth_100/
│   ├── sydney_49/
│   └── sydney_100/
└── src/                   # Main execution scripts.
```

---

## 🚀 Getting Started

To replicate our experiments, please follow the steps below.

### Prerequisites

The experiments were run on a system with the following specifications:
* **Python:** `3.10.12` (we recommend using a virtual environment like `venv`).
* **GPU:** NVIDIA GPU with CUDA `12.2`.
* **NVIDIA Driver:** `532.183.01`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mariomttz/wec-optimization.git
    cd wec-optimization
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .env
    source .env/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install optuna fastai torch catboost
    ```
> [!NOTE]
> This command will also install dependencies such as `NumPy`, `Scikit-learn`, and `Pandas`.

---

## ⚡ Running the Experiments

The experiments are orchestrated using shell scripts located in the `src/` directory.

### Environment Configuration

Before running the scripts, you must set the following environment variables. These are used to send notifications via Telegram upon completion.

```bash
export BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
export CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
export PROJECT_ROOT="/path/to/your/wec-optimization"
```

> [!WARNING]
> Ensure the `PROJECT_ROOT` variable points to the absolute path of the cloned repository directory.

### Execution

1.  **Run the Machine Learning Models Training:**
    This script will train all four models on all four datasets.
    ```bash
    cd src/
    bash run_all_models.sh
    ```

2.  **Run the Optimization Algorithms:**
    This script will run the DE and PSO algorithms to find the optimal WEC configurations using the best-trained surrogate models.
    ```bash
    cd src/
    bash run_all_optimizations.sh
    ```

---

## 📊 Results

Our approach significantly outperformed the results from the original study that relied on a hydrodynamic simulator. The use of surrogate models allowed for a much more exhaustive exploration of the solution space.

| Subset      | Best Result in the Original Paper | Our Best Result (DE + SVM/CatBoost) | Improvement  |
|-------------|-----------------------------------|-------------------------------------|--------------|
| Perth 49    | 4.18 MW                           | **4.56 MW**                         | +9.09%       |
| Perth 100   | 7.35 MW                           | **10.11 MW**                        | +37.55%      |
| Sydney 49   | 4.15 MW                           | **4.47 MW**                         | +7.71%       |
| Sydney 100  | 7.36 MW                           | **7.90 MW**                         | +7.34%       |

_Comparison of maximum power output (MW)._

---

## 🔗 References & Citation

This work builds upon the foundational research by Neshat et al. and utilizes their public dataset.

* **Original Paper:** [Optimisation of large wave farms using a multi-strategy evolutionary framework](https://dl.acm.org/doi/10.1145/3377930.3390235)
* **Dataset:** [Large-scale Wave Energy Farm (UCI)](https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm)

If you use this code or our findings in your research, please consider citing our paper (citation details to be added upon publication).

---

## 📧 Contact

* **Mario A. Martinez Oliveros:** `mttzoma@gmail.com`
* **Dr. Adriana Menchaca Méndez:** `amenchaca@enesmorelia.unam.mx`
* **Dr. Miguel Raggi Pérez:** `mraggi@enesmorelia.unam.mx`
