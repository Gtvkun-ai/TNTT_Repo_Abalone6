# Experiments

Thu muc nay dung de chay cac thu nghiem rieng biet va luu cach so sanh giua cac bien the mo hinh.

## Cau hinh dang ap dung

- Ty le chia du lieu: 70% train, 30% test
- Random seed: 42
- So fold cross-validation theo yeu cau do an: 5
- Ket qua duoc luu duoi dang JSON de de dua vao bao cao

## Danh sach file

- `01_baseline_random_forest.py`: baseline don gian voi RandomForestRegressor.
- `02_preprocessing_comparison.py`: so sanh ba cach tien xu ly.
- `03_feature_engineering_ablation.py`: so sanh giua du lieu goc va du lieu co feature engineering.
- `_common.py`: cau hinh va ham dung chung cho cac script.
- `results/`: noi luu file JSON ket qua cho tung lan thu nghiem.

## Cach chay

Su dung duong dan tuyet doi:

```bash
python C:/Users/Gtvkun/Documents/GitHub/TNTT_Repo_NhomYenHoa/AbaloneAge/experiments/01_baseline_random_forest.py
python C:/Users/Gtvkun/Documents/GitHub/TNTT_Repo_NhomYenHoa/AbaloneAge/experiments/02_preprocessing_comparison.py
python C:/Users/Gtvkun/Documents/GitHub/TNTT_Repo_NhomYenHoa/AbaloneAge/experiments/03_feature_engineering_ablation.py
```

## Vi tri luu ket qua

- `experiments/results/`
- `outputs/metrics/experiments/`

## Ghi chu

- Cac script dung du lieu `data/raw/abalone.csv`.
- Bo script nay la khung thu nghiem de tong hop phan `Experiments and Discussion` trong bao cao.
- Hien tai metadata da ghi dung ty le 70/30;
