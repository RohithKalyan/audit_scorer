import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool

# Load CP Weights (only static input allowed)
# Directly embed CP scores
cp_score_dict = {
    "CP_01": 83, "CP_02": 86, "CP_03": 78, "CP_04": 81, "CP_07": 84, "CP_08": 80,
    "CP_09": 76, "CP_10": 79, "CP_11": 82, "CP_13": 77, "CP_14": 85, "CP_15": 88,
    "CP_16": 73, "CP_17": 75, "CP_18": 90, "CP_19": 60, "CP_20": 74, "CP_21": 69,
    "CP_22": 66, "CP_23": 87, "CP_24": 78, "CP_26": 0,  "CP_27": 71, "CP_28": 67,
    "CP_30": 72, "CP_31": 70, "CP_32": 72
}



def score_uploaded_file(df):
    df = df.copy()


    # === Step 1: CONTROL POINTS ===
    desc_col = 'Description'
    ref_col = 'Reference'
    net_col = 'Net'
    date_col = 'Date'
    acc_cat_col = 'GL Account Category'

    keywords = ['fraud','bribe','kickback','suspicious','fake','dummy','gift','prize','token','free','reward','favour']
    df["CP_01"] = df[desc_col].fillna("").str.lower().apply(lambda x: int(any(k in x for k in keywords)))
    df["CP_02"] = (df[net_col].abs() > df[net_col].abs().quantile(0.98)).astype(int)
    df["CP_03"] = df.duplicated(subset=[date_col, ref_col, net_col], keep=False).astype(int)
    sales_mask = df[acc_cat_col].astype(str).str.lower().str.contains("sales", na=False)
    materiality = 0.005 * df.loc[sales_mask, net_col].abs().sum()
    df["CP_04"] = (sales_mask & (df[net_col].abs() > materiality)).astype(int)
    df["CP_08"] = (
        df[acc_cat_col].astype(str).str.lower().str.contains('cash', na=False) |
        df[desc_col].astype(str).str.lower().str.contains('cash', na=False)
    ).astype(int)
    df["CP_09"] = (
        df[acc_cat_col].astype(str).str.lower().str.contains('bad debt', na=False) &
        df[desc_col].astype(str).str.lower().str.contains('cash', na=False)
    ).astype(int)
    df["CP_10"] = (df[net_col].abs() > 2 * df[net_col].std()).astype(int)
    outlier_thresh = df[net_col].abs().mean() + 3 * df[net_col].abs().std()
    df["CP_11"] = (df[net_col].abs() > outlier_thresh).astype(int)
    related_terms = ['related','subsidiary','affiliate','group company','holding']
    df["CP_14"] = df[desc_col].fillna("").str.lower().apply(lambda x: int(any(k in x for k in related_terms)))
    split_grp = df.groupby([date_col, ref_col, desc_col])[net_col].transform('sum')
    split_count = df.groupby([date_col, ref_col, desc_col])[net_col].transform('count')
    df["CP_15"] = ((split_count > 1) & (split_grp > materiality)).astype(int)
    df["CP_16"] = 0
    vendor_col = desc_col
    vendor_share = df[vendor_col].value_counts(normalize=True)
    df["CP_17"] = df[vendor_col].isin(vendor_share[vendor_share > 0.05].index).astype(int)
    df["CP_19"] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True).dt.weekday.isin([5,6]).astype(int)
    df["CP_20"] = df[desc_col].isna() | df[desc_col].astype(str).str.strip().eq("")
    parsed_dates = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df["CP_21"] = (parsed_dates == parsed_dates.max()).astype(int)
    df["CP_22"] = (parsed_dates == parsed_dates.min()).astype(int)
    complex_terms = ['derivative','spv','structured','note','swap']
    df["CP_23"] = (
        df[desc_col].astype(str).str.lower().apply(lambda x: int(any(t in x for t in complex_terms))) |
        df[acc_cat_col].astype(str).str.lower().apply(lambda x: int(any(t in x for t in complex_terms)))
    ).astype(int)
    seqs = {'123','234','345','456','567','678','789','890','321','432','543','654','765','876','987','098'}
    repeats = {str(i)*3 for i in range(10)}
    def check_last3(n):
        try:
            last = str(int(abs(n)))[-3:]
            return int(last in seqs or last in repeats and last != '901')
        except: return 0
    df["CP_24"] = df[net_col].apply(check_last3)
    def is_first_weekend(date_str):
        try:
            d = pd.to_datetime(date_str, dayfirst=True)
            for i in range(7):
                first_day = d.replace(day=1)
                test_day = first_day + pd.Timedelta(days=i)
                if test_day.weekday() in [5,6]:
                    return int(d.date() == test_day.date())
        except: return 0
    df["CP_27"] = df[date_col].apply(is_first_weekend)
    inst = ['derivative','option','swap','future','structured']
    df["CP_30"] = (
        df[desc_col].astype(str).str.lower().apply(lambda x: int(any(i in x for i in inst))) |
        df[acc_cat_col].astype(str).str.lower().apply(lambda x: int(any(i in x for i in inst)))
    ).astype(int)
    df["Month_CP31"] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True).dt.to_period("M")
    counts = df.groupby("Month_CP31").size()
    flag_months = set(counts[counts.pct_change().abs() > 1.0].index)
    df["CP_31"] = df["Month_CP31"].isin(flag_months).astype(int)
    df.drop(columns="Month_CP31", inplace=True)
    df["CP_32"] = (df[net_col] == 0).astype(int)

    # === Step 2: CP Score Calculation ===
    valid_cps = [f"CP_{i:02}" for i in range(1, 33) if i not in [5, 6, 12, 25]]
    def compute_cp_scores(row):
        triggered = []
        cp_probs = []
        for cp in valid_cps:
            if row.get(cp, 0) == 1:
                score = cp_score_dict.get(cp, 0)
                triggered.append(f"{cp} ({score})")
                cp_probs.append(1 - score / 100)
        score = 1 - np.prod(cp_probs) if cp_probs else 0
        return pd.Series({"Triggered_CPs": ", ".join(triggered), "CP_Score": round(score, 4)})
    df = df.join(df.apply(compute_cp_scores, axis=1))

    # === Step 3: Narration Risk Model (TF-IDF) ===
    df["Line Desc"] = df[desc_col].fillna("")
    y_train = df["CP_01"]
    if y_train.nunique() >= 2:
        tfidf = TfidfVectorizer(max_features=200)
        X_text = tfidf.fit_transform(df["Line Desc"])
        logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
        logreg.fit(X_text, y_train)
        df["narration_risk_score"] = logreg.predict_proba(tfidf.transform(df["Line Desc"]))[:, 1]
    else:
        df["narration_risk_score"] = 0.0

    # === Step 4: CatBoost Model ===
    labels = (df["CP_Score"] > 0.6).astype(int)
    exclude_cols = valid_cps + [
        "S. No", "Risk Level", "Risk",
        "Line Desc", "narration_risk_score",
        "Triggered_CPs", "CP_Score", "Model_Score", "Final_Score"
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    cat_cols = [col for col in features if df[col].dtype == "object"]
    num_cols = [col for col in features if col not in cat_cols]
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
    for col in num_cols:
        df[col] = df[col].fillna(0)
    train_pool = Pool(data=df[cat_cols + num_cols], label=labels, cat_features=cat_cols)
    cb_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, class_weights=[1, 3], verbose=0)
    cb_model.fit(train_pool)
    df["Model_Score"] = cb_model.predict_proba(train_pool)[:, 1]

    # === Step 5: Final Score ===
    df["Final_Score"] = (0.6 * df["CP_Score"] + 0.4 * df["Model_Score"]).round(4)

    return df[["Date", "Triggered_CPs", "CP_Score", "Line Desc", "narration_risk_score", "Model_Score", "Final_Score"]]
