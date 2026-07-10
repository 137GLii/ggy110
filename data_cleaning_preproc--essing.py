"""
数据清洗与预处理脚本
====================
功能：
  1. 读取包含缺失值和异常值的数据集
  2. 实现多种缺失值处理方法（删除、填充、插值）
  3. 识别并处理重复记录
  4. 进行数据类型转换和格式标准化

数据集：使用 Kaggle Titanic 数据集作为示例，同时支持房价预测数据集
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore")


# ============================================================================
# 第1步：读取数据
# ============================================================================
def load_data(source="titanic"):
    """
    读取数据集。支持 Titanic 和 Housing 两种数据集。
    如果没有本地文件，则自动生成模拟数据用于演示。
    """
    if source == "titanic":
        url = ("https://raw.githubusercontent.com/datasciencedojo/"
               "datasets/master/titanic.csv")
        try:
            df = pd.read_csv(url)
            print("[INFO] 成功从网络加载 Titanic 数据集")
        except Exception:
            print("[WARN] 无法连接网络，生成模拟 Titanic 数据")
            df = generate_mock_titanic_data()
    elif source == "housing":
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["MedHouseVal"] = data.target
            print("[INFO] 加载 California Housing 数据集")
        except Exception:
            print("[WARN] 生成模拟房价数据")
            df = generate_mock_housing_data()
    else:
        raise ValueError(f"未知的数据源: {source}")

    return df


def generate_mock_titanic_data(n_rows=891):
    """生成模拟的 Titanic 数据集（用于离线演示）"""
    np.random.seed(42)
    df = pd.DataFrame({
        "PassengerId": range(1, n_rows + 1),
        "Survived": np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
        "Pclass": np.random.choice([1, 2, 3], n_rows, p=[0.24, 0.21, 0.55]),
        "Name": [f"Passenger_{i}" for i in range(1, n_rows + 1)],
        "Sex": np.random.choice(["male", "female"], n_rows),
        "Age": np.random.normal(30, 14, n_rows),
        "SibSp": np.random.choice([0, 1, 2, 3, 4, 5, 8], n_rows,
                                  p=[0.68, 0.23, 0.03, 0.02, 0.02, 0.01, 0.01]),
        "Parch": np.random.choice([0, 1, 2, 3, 4, 5, 6], n_rows,
                                  p=[0.76, 0.13, 0.09, 0.02, 0.0, 0.0, 0.0]),
        "Ticket": [f"TICKET-{i}" for i in range(1, n_rows + 1)],
        "Fare": np.random.uniform(0, 512, n_rows),
        "Cabin": np.random.choice([f"C{np.random.randint(1,150)}", np.nan],
                                  n_rows, p=[0.23, 0.77]),
        "Embarked": np.random.choice(["S", "C", "Q", np.nan],
                                     n_rows, p=[0.72, 0.19, 0.08, 0.01]),
    })
    # 人为注入更多缺失值
    mask_age = np.random.random(n_rows) < 0.20
    df.loc[mask_age, "Age"] = np.nan
    # 人为注入异常值
    outlier_idx = np.random.choice(n_rows, 15, replace=False)
    df.loc[outlier_idx, "Fare"] = np.random.uniform(1000, 5000, 15)
    # 重复记录
    duplicates = df.sample(n=5, random_state=99)
    df = pd.concat([df, duplicates], ignore_index=True)
    # 格式不统一
    df.loc[df.sample(10, random_state=7).index, "Sex"] = ["MALE", "FEMALE"] * 5
    df.loc[df.sample(5, random_state=3).index, "Embarked"] = ["s", "c", "q", "s", "c"]
    return df


def generate_mock_housing_data(n_rows=500):
    """生成模拟房价数据"""
    np.random.seed(42)
    df = pd.DataFrame({
        "MedInc": np.random.uniform(0.5, 15, n_rows),
        "HouseAge": np.random.uniform(1, 52, n_rows),
        "AveRooms": np.random.uniform(2, 10, n_rows),
        "AveBedrms": np.random.uniform(1, 5, n_rows),
        "Population": np.random.uniform(100, 30000, n_rows),
        "AveOccup": np.random.uniform(1, 6, n_rows),
        "Latitude": np.random.uniform(32, 42, n_rows),
        "Longitude": np.random.uniform(-124, -114, n_rows),
        "MedHouseVal": np.random.uniform(0.15, 5, n_rows),
    })
    # 注入缺失值
    for col in ["MedInc", "HouseAge", "AveRooms"]:
        df.loc[np.random.random(n_rows) < 0.10, col] = np.nan
    # 异常值
    df.loc[np.random.choice(n_rows, 10, replace=False), "AveOccup"] = 100
    # 重复
    dup = df.sample(5, random_state=7)
    df = pd.concat([df, dup], ignore_index=True)
    return df


# ============================================================================
# 第2步：数据探索与概览
# ============================================================================
def explore_data(df):
    """对数据集进行初步探索"""
    print("\n" + "=" * 60)
    print("📊 数据概览")
    print("=" * 60)
    print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"\n列名及数据类型:\n{df.dtypes}\n")

    print("-" * 40)
    print("缺失值统计:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"缺失数量": missing, "缺失比例(%)": missing_pct})
    print(missing_df[missing_df["缺失数量"] > 0])

    print("\n" + "-" * 40)
    print("重复行数量:", df.duplicated().sum())

    print("\n" + "-" * 40)
    print("数值列统计描述:")
    print(df.describe().round(2))

    return missing_df


# ============================================================================
# 第3步：缺失值处理
# ============================================================================
def handle_missing_values(df, strategy="comprehensive"):
    """
    实现多种缺失值处理方法：
      - 删除法: 删除含缺失值的行 / 删除缺失比例过高的列
      - 填充法: 均值/中位数/众数填充、常量填充、前向/后向填充
      - 插值法: 线性插值、多项式插值、KNN插值、多重插值
    """
    df_clean = df.copy()
    print("\n" + "=" * 60)
    print("🔧 缺失值处理")
    print("=" * 60)

    # --- 3.1 删除法 ---
    print("\n--- 3.1 删除法 ---")

    # 删除缺失比例超过阈值的列 (如 Titanic 的 Cabin 列)
    threshold = 0.70
    cols_before = set(df_clean.columns)
    missing_ratio = df_clean.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        print(f"删除缺失比例 > {threshold:.0%} 的列: {cols_to_drop}")

    # 删除含缺失值的行（仅对关键列）
    before_rows = len(df_clean)
    key_cols = [c for c in df_clean.columns
                if df_clean[c].dtype in [np.float64, np.int64]]
    if key_cols:
        df_clean.dropna(subset=key_cols[:3], inplace=True)
        print(f"基于关键列删除含缺失行: {before_rows} → {len(df_clean)} 行 "
              f"(删除 {before_rows - len(df_clean)} 行)")

    # --- 3.2 填充法 ---
    print("\n--- 3.2 填充法 ---")

    # 分离数值列与非数值列
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    # 数值列：中位数填充
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  [数值-中位数填充] {col}: 填充 {int(df[col].isnull().sum())} 个缺失值 → 中位数 {median_val:.2f}")

    # 类别列：众数填充
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df_clean[col].fillna(fill_val, inplace=True)
            print(f"  [类别-众数填充] {col}: 填充为 '{fill_val}'")

    # --- 3.3 高级插值法 (演示用) ---
    print("\n--- 3.3 高级插值法 (对比) ---")

    # 线性插值
    df_interp_linear = df[num_cols].interpolate(method="linear", limit_direction="both")
    linear_filled = df_interp_linear.isnull().sum().sum() - df_clean[num_cols].isnull().sum().sum()
    print(f"  线性插值: 可处理 {int(abs(linear_filled))} 个额外缺失值")

    # 多项式插值
    try:
        df_interp_poly = df[num_cols].interpolate(method="polynomial", order=2,
                                                   limit_direction="both")
        poly_filled = df_interp_poly.isnull().sum().sum() - df_clean[num_cols].isnull().sum().sum()
        print(f"  多项式插值 (order=2): 可处理 {int(abs(poly_filled))} 个额外缺失值")
    except Exception:
        print("  多项式插值: 不适用于当前数据")

    # KNN 插值
    if len(num_cols) >= 2:
        try:
            knn_imputer = KNNImputer(n_neighbors=5)
            df_knn = pd.DataFrame(
                knn_imputer.fit_transform(df[num_cols]),
                columns=num_cols
            )
            print(f"  KNN 插值 (k=5): 完成 (基于最近邻的多维估算)")
        except Exception as e:
            print(f"  KNN 插值: 失败 ({e})")

    # 多重插值 (IterativeImputer - MICE)
    if len(num_cols) >= 2:
        try:
            mice_imputer = IterativeImputer(max_iter=10, random_state=42)
            df_mice = pd.DataFrame(
                mice_imputer.fit_transform(df[num_cols]),
                columns=num_cols
            )
            print(f"  多重插值 (MICE, iter=10): 完成 (基于迭代回归的估计)")
        except Exception as e:
            print(f"  多重插值: 失败 ({e})")

    print(f"\n最终剩余缺失值总数: {df_clean.isnull().sum().sum()}")
    return df_clean


# ============================================================================
# 第4步：异常值识别与处理
# ============================================================================
def handle_outliers(df):
    """
    识别并处理异常值：
      - IQR 方法识别异常值
      - Z-score 方法识别异常值
      - 使用 capping（盖帽法）处理异常值
    """
    df_clean = df.copy()
    print("\n" + "=" * 60)
    print("🎯 异常值识别与处理")
    print("=" * 60)

    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # 排除 ID 类列
    exclude = [c for c in num_cols if "id" in c.lower()]
    num_cols = [c for c in num_cols if c not in exclude]

    outlier_report = {}

    for col in num_cols:
        # --- IQR 方法 ---
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((df_clean[col] < lower_bound) |
                        (df_clean[col] > upper_bound)).sum()

        # --- Z-score 方法 ---
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) /
                          df_clean[col].std())
        z_outliers = (z_scores > 3).sum()

        if iqr_outliers > 0 or z_outliers > 0:
            outlier_report[col] = {
                "IQR异常值": iqr_outliers,
                "Z-score异常值(>3σ)": z_outliers,
                "下界": round(lower_bound, 2),
                "上界": round(upper_bound, 2),
            }

        # --- 盖帽法处理 (Capping) ---
        if iqr_outliers > 0:
            df_clean[col] = df_clean[col].clip(lower=lower_bound,
                                                upper=upper_bound)

    if outlier_report:
        print("\n异常值检测报告:")
        for col, info in outlier_report.items():
            print(f"\n  📌 {col}:")
            for k, v in info.items():
                print(f"      {k}: {v}")
        print("\n处理方式: 盖帽法 (Winsorizing) — 超出 [Q1-1.5×IQR, Q3+1.5×IQR] 的值被截断")
    else:
        print("未检测到显著异常值。")

    return df_clean


# ============================================================================
# 第5步：重复记录处理
# ============================================================================
def handle_duplicates(df):
    """识别并处理重复记录"""
    print("\n" + "=" * 60)
    print("📋 重复记录处理")
    print("=" * 60)

    df_clean = df.copy()

    # 检测完全重复的行
    full_dup = df_clean.duplicated().sum()
    print(f"完全重复行: {full_dup}")

    if full_dup > 0:
        # 显示重复行
        dup_mask = df_clean.duplicated(keep=False)
        print(f"\n重复记录详情 (前10行):")
        print(df_clean[dup_mask].head(10))
        # 删除重复（保留第一次出现）
        df_clean.drop_duplicates(keep="first", inplace=True)
        print(f"\n已删除 {full_dup} 行重复记录，保留首次出现。"
              f"新形状: {df_clean.shape}")

    # 检测基于子集的重复（如相同乘客信息）
    subset_cols = [c for c in df_clean.columns
                   if c in ["Name", "Age", "Sex", "Ticket"]]
    if len(subset_cols) >= 2:
        subset_dup = df_clean.duplicated(subset=subset_cols).sum()
        print(f"基于 {subset_cols} 的部分重复: {subset_dup}")

    return df_clean


# ============================================================================
# 第6步：数据类型转换与格式标准化
# ============================================================================
def standardize_data(df):
    """
    数据类型转换与格式标准化：
      - 类别数据规范化（大小写统一、空白去除）
      - 数值类型转换
      - 日期格式标准化
      - 标签编码
    """
    print("\n" + "=" * 60)
    print("📐 数据类型转换与格式标准化")
    print("=" * 60)

    df_clean = df.copy()

    # --- 6.1 字符串列规范化 ---
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in str_cols:
        # 去除首尾空白
        df_clean[col] = df_clean[col].str.strip()
        # 统一小写（类别数据）
        if df_clean[col].nunique() < 20:
            df_clean[col] = df_clean[col].str.lower()
            print(f"  [规范化] {col}: 统一为小写，{df_clean[col].nunique()} 个唯一值")
        else:
            print(f"  [规范化] {col}: 去除首尾空白")

    # --- 6.2 数值类型转换 ---
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            # 尝试转换为数值
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
                print(f"  [类型转换] {col}: object → numeric")
            except (ValueError, TypeError):
                pass

    # 将合适的整数列转换类型
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col.lower() == "survived" or "class" in col.lower():
            df_clean[col] = df_clean[col].astype(int)
            print(f"  [类型转换] {col}: → int")

    # --- 6.3 类别编码 ---
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df_clean[col].nunique() <= 10:
            le = LabelEncoder()
            encoded_col = le.fit_transform(df_clean[col].astype(str))
            print(f"  [标签编码] {col}: {df_clean[col].nunique()} 个类别 → "
                  f"0~{df_clean[col].nunique() - 1}")
            df_clean[col + "_Encoded"] = encoded_col

    # --- 6.4 打印最终数据类型 ---
    print(f"\n标准化后的数据类型:")
    print(df_clean.dtypes)

    return df_clean


# ============================================================================
# 第7步：综合报告
# ============================================================================
def generate_report(original_df, cleaned_df):
    """生成数据清洗前后对比报告"""
    print("\n" + "=" * 60)
    print("📋 综合清洗报告")
    print("=" * 60)

    print(f"\n{'指标':<30} {'清洗前':>10} {'清洗后':>10}")
    print("-" * 52)
    print(f"{'行数':<30} {original_df.shape[0]:>10} {cleaned_df.shape[0]:>10}")
    print(f"{'列数':<30} {original_df.shape[1]:>10} {cleaned_df.shape[1]:>10}")
    print(f"{'缺失值总数':<30} "
          f"{original_df.isnull().sum().sum():>10} "
          f"{cleaned_df.isnull().sum().sum():>10}")
    print(f"{'重复行':<30} "
          f"{original_df.duplicated().sum():>10} "
          f"{cleaned_df.duplicated().sum():>10}")
    print(f"{'数值列数':<30} "
          f"{original_df.select_dtypes(include=[np.number]).shape[1]:>10} "
          f"{cleaned_df.select_dtypes(include=[np.number]).shape[1]:>10}")
    print(f"{'类别列数':<30} "
          f"{original_df.select_dtypes(include=['object']).shape[1]:>10} "
          f"{cleaned_df.select_dtypes(include=['object']).shape[1]:>10}")

    # 清洗建议
    print("\n" + "-" * 40)
    print("✅ 已完成的清洗步骤:")
    steps = [
        "1. 数据加载与初步探索",
        "2. 缺失值检测 → 删除法 + 填充法 + 插值法",
        "3. 异常值识别 (IQR + Z-score) → 盖帽法处理",
        "4. 重复记录检测与删除",
        "5. 数据类型转换",
        "6. 字符串格式标准化",
        "7. 类别标签编码",
    ]
    for step in steps:
        print(f"   {step}")

    return cleaned_df


# ============================================================================
# 主流程
# ============================================================================
def main(source="titanic"):
    """主流程：依次执行所有清洗步骤"""
    print("=" * 60)
    print("🚀 数据清洗与预处理流程")
    print("=" * 60)

    # 第1步：读取数据
    df_raw = load_data(source)
    print(f"[OK] 数据加载完成: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

    # 第2步：数据探索
    explore_data(df_raw)

    # 第3步：缺失值处理
    df_clean = handle_missing_values(df_raw)
    print(f"[OK] 缺失值处理完成")

    # 第4步：异常值处理
    df_clean = handle_outliers(df_clean)
    print(f"[OK] 异常值处理完成")

    # 第5步：重复记录处理
    df_clean = handle_duplicates(df_clean)
    print(f"[OK] 重复记录处理完成")

    # 第6步：格式标准化
    df_clean = standardize_data(df_clean)
    print(f"[OK] 格式标准化完成")

    # 第7步：综合报告
    generate_report(df_raw, df_clean)

    # 保存清洗后数据
    output_path = f"cleaned_{source}_data.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n💾 清洗后数据已保存至: {output_path}")

    return df_raw, df_clean


if __name__ == "__main__":
    import sys

    # 可通过命令行参数选择数据集: python script.py titanic 或 python script.py housing
    source = sys.argv[1] if len(sys.argv) > 1 else "titanic"
    raw, clean = main(source)
