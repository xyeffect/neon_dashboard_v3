"""
NEON Enrollment Forecasting Pipeline
Modules 1, 2, 3

Input files:
    school_performance_analysis_updated.csv
    drop_timeline_analysis_updated.csv
    New_Course_Allocation.csv
    school.csv
    course.csv
    26_03_20_Cornell PiTech File  Student Enrollment.csv

Output files:
    school_risk_profile.csv
    ca_enrollment_predictions.csv
    school_enrollment_model.pkl
    course_thresholds.csv
    sst_intervention_flags.csv
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import adjusted_rand_score, mean_absolute_error

warnings.filterwarnings("ignore")

# =============================================================================
# MODULE 1 — School Risk Classification
# =============================================================================
print("\n" + "="*60)
print("MODULE 1 — School Risk Classification")
print("="*60)

perf   = pd.read_csv("school_performance_analysis_updated.csv")
drops  = pd.read_csv("drop_timeline_analysis_updated.csv")
ca     = pd.read_csv("New_Course_Allocations.csv")
sch    = pd.read_csv("New_Schools.csv")
enroll = pd.read_csv("Student Enrollment.csv",
                     parse_dates=["Created Date"])
course = pd.read_csv("course.csv", parse_dates=["Start Date"])

# Archived = course completed normally, not a drop
TRUE_DROPS = ["Admissions Drop", "Dropped"]

# -----------------------------------------------------------------------------
# Feature 1: Drop Rate
# -----------------------------------------------------------------------------
ca["is_true_drop"] = ca["End Status"].isin(TRUE_DROPS).astype(int)

school_dr = (
    ca.groupby("School Name")
    .agg(
        total_ca  = ("Course Allocation ID", "count"),
        n_dropped = ("is_true_drop", "sum"),
    )
    .reset_index()
)
school_dr["drop_rate"] = school_dr["n_dropped"] / school_dr["total_ca"]

# -----------------------------------------------------------------------------
# Feature 2: Late Drop Rate
# -----------------------------------------------------------------------------
LATE_DROP_DAYS = 30
drops["is_late_drop"] = (drops["Days_Before_Start"] <= LATE_DROP_DAYS).astype(int)

late_dr = (
    drops.groupby("School Name")
    .agg(
        n_dropped_total  = ("is_late_drop", "count"),
        n_late           = ("is_late_drop", "sum"),
        avg_days_to_drop = ("Days_Before_Start", "mean"),
    )
    .reset_index()
)
late_dr["late_drop_rate"] = late_dr["n_late"] / late_dr["n_dropped_total"].clip(lower=1)

# -----------------------------------------------------------------------------
# Feature 3: Low Utilization Rate (Squat Rate)
# -----------------------------------------------------------------------------
ca["is_active"]    = (~ca["End Status"].isin(TRUE_DROPS)).astype(int)
ca["is_squatting"] = (
    (ca["is_active"] == 1) & (ca["Enrollment Count"] < 5)
).astype(int)

# -----------------------------------------------------------------------------
# Feature 4: Avg Participating per CA and Enrollment CV
# -----------------------------------------------------------------------------
school_enroll = (
    ca.groupby("School Name")
    .agg(
        squat_rate   = ("is_squatting",     "mean"),
        avg_enrolled = ("Enrollment Count", "mean"),
        std_enrolled = ("Enrollment Count", "std"),
        n_ca_total   = ("Course Allocation ID", "count"),
    )
    .reset_index()
)
school_enroll["enroll_cv"] = (
    school_enroll["std_enrolled"] / (school_enroll["avg_enrolled"] + 1e-6)
)

# -----------------------------------------------------------------------------
# Feature 5: LEA Partner Flag
# -----------------------------------------------------------------------------
lea = (
    sch[["School Name", "LEA Name"]]
    .drop_duplicates("School Name")
    .assign(
        is_lea_partner=lambda d: d["LEA Name"].notna() & (d["LEA Name"].str.strip() != "")
    )
    [["School Name", "is_lea_partner"]]
)

# -----------------------------------------------------------------------------
# Feature 6: Late Enrollment Rate
# Share of students who enrolled within 30 days of course start.
# Uses actual student enrollment dates matched to course start dates.
# Negative days_before_start (data entry after course start) are excluded.
# -----------------------------------------------------------------------------
LATE_ENROLL_DAYS = 30

course_start = (
    course[course["Start Date"].notna()]
    [["Course Name", "Term Name", "Start Date"]]
    .drop_duplicates()
)

enroll_merged = enroll.merge(
    course_start,
    left_on=["Course Name", "Term"],
    right_on=["Course Name", "Term Name"],
    how="left"
)

enroll_merged["days_before_start"] = (
    enroll_merged["Start Date"] - enroll_merged["Created Date"]
).dt.days

# Exclude records where enrollment was recorded after course start
enroll_valid = enroll_merged[enroll_merged["days_before_start"] >= 0].copy()
enroll_valid["is_late_enroll"] = (
    enroll_valid["days_before_start"] <= LATE_ENROLL_DAYS
).astype(int)

late_enroll = (
    enroll_valid.groupby("School Name")
    .agg(
        total_enrollments = ("is_late_enroll", "count"),
        late_enrollments  = ("is_late_enroll", "sum"),
        late_enroll_rate  = ("is_late_enroll", "mean"),
    )
    .reset_index()
)

print(f"Schools with late enroll data: {len(late_enroll)}")
print(late_enroll["late_enroll_rate"].describe().round(3))

# -----------------------------------------------------------------------------
# Merge all features
# -----------------------------------------------------------------------------
feat = (
    school_dr
    .merge(late_dr[["School Name", "late_drop_rate", "avg_days_to_drop"]],
           on="School Name", how="left")
    .merge(school_enroll[["School Name", "squat_rate", "avg_enrolled",
                           "enroll_cv", "n_ca_total"]],
           on="School Name", how="left")
    .merge(lea, on="School Name", how="left")
    .merge(late_enroll[["School Name", "late_enroll_rate"]],
           on="School Name", how="left")
)
feat.fillna({
    "late_drop_rate":   0,
    "squat_rate":       0,
    "enroll_cv":        0,
    "is_lea_partner":   False,
    "late_enroll_rate": 0,
}, inplace=True)
feat["avg_enrolled"] = feat["avg_enrolled"].fillna(feat["avg_enrolled"].median())

# -----------------------------------------------------------------------------
# Rule-Based Tiering
# -----------------------------------------------------------------------------
HIGH_DROP      = 0.60
HIGH_LATE_DROP = 0.40
HIGH_SQUAT     = 0.25
LOW_ENROLL     = 6
HIGH_CV        = 0.90
MIN_OBS_LOW    = 3

def assign_risk_tier(row):
    if row["drop_rate"] >= 0.85:
        return "High"
    if row["drop_rate"] >= HIGH_DROP and row["avg_enrolled"] < LOW_ENROLL:
        return "High"
    if row["drop_rate"] >= HIGH_DROP and row["enroll_cv"] >= HIGH_CV:
        return "High"
    signals = sum([
        row["drop_rate"]      >= HIGH_DROP,
        row["late_drop_rate"] >= HIGH_LATE_DROP,
        row["squat_rate"]     >= HIGH_SQUAT,
        row["avg_enrolled"]   <  LOW_ENROLL,
        row["enroll_cv"]      >= HIGH_CV,
    ])
    tier = "High" if signals >= 2 else ("Medium" if signals == 1 else "Low")
    if tier == "Low" and row["total_ca"] < MIN_OBS_LOW:
        if not (row["drop_rate"] == 0 and row["avg_enrolled"] >= 8):
            return "Medium"
    return tier

feat["risk_rule"] = feat.apply(assign_risk_tier, axis=1)
print("\nRule-based tier distribution:")
print(feat["risk_rule"].value_counts())

# -----------------------------------------------------------------------------
# K-Means Validation
# -----------------------------------------------------------------------------
CLUSTER_FEATURES = ["drop_rate","late_drop_rate","squat_rate","avg_enrolled","enroll_cv"]
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat[CLUSTER_FEATURES].fillna(0))

km = KMeans(n_clusters=3, random_state=42, n_init=10)
feat["cluster"] = km.fit_predict(X_scaled)
cluster_order   = feat.groupby("cluster")["drop_rate"].mean().sort_values()
cluster_map     = {
    cluster_order.index[0]: "Low",
    cluster_order.index[1]: "Medium",
    cluster_order.index[2]: "High",
}
feat["risk_cluster"] = feat["cluster"].map(cluster_map)

ari = adjusted_rand_score(feat["risk_rule"], feat["risk_cluster"])
print(f"\nARI (rule vs cluster): {ari:.3f}")
print(pd.crosstab(feat["risk_rule"], feat["risk_cluster"], margins=True))

# -----------------------------------------------------------------------------
# Attrition and Survival Rates
# -----------------------------------------------------------------------------
feat["risk_final"]     = feat["risk_rule"]
attrition              = feat.groupby("risk_final")["drop_rate"].mean()
feat["attrition_rate"] = feat["risk_final"].map(attrition)
feat["survival_rate"]  = 1 - feat["attrition_rate"]

print("\nAttrition rates by tier:")
print(attrition.round(3))

out_cols_1 = [
    "School Name", "total_ca", "drop_rate", "late_drop_rate",
    "avg_days_to_drop", "squat_rate", "avg_enrolled", "enroll_cv",
    "is_lea_partner", "risk_final", "risk_cluster",
    "attrition_rate", "survival_rate", "late_enroll_rate",
]
feat[out_cols_1].rename(columns={
    "drop_rate":        "ca_drop_rate",
    "late_drop_rate":   "ca_late_drop_rate",
    "squat_rate":       "ca_low_utilization_rate",
    "avg_enrolled":     "avg_participating_per_ca",
    "enroll_cv":        "school_enrollment_cv",
}).to_csv("school_risk_profile.csv", index=False)
print(f"\nSaved → school_risk_profile.csv ({len(feat)} schools)")


# =============================================================================
# MODULE 2 — Enrollment Prediction
# =============================================================================
print("\n" + "="*60)
print("MODULE 2 — Enrollment Prediction")
print("="*60)

risk = pd.read_csv("school_risk_profile.csv")

def get_season(term):
    t = str(term).strip()
    if "Fall"   in t: return "Fall"
    if "Spring" in t: return "Spring"
    return "Unknown"

ca["season"] = ca["Term Name"].apply(get_season)

course_meta = (
    course[["Course Name", "Subject", "Seat Goal", "Seat Capacity"]]
    .drop_duplicates("Course Name")
)
ca = ca.merge(course_meta, on="Course Name", how="left")
ca = ca.merge(
    risk[["School Name", "risk_final", "avg_participating_per_ca", "school_enrollment_cv",
          "ca_drop_rate", "survival_rate", "is_lea_partner"]],
    on="School Name", how="left",
)

school_course_avg = (
    ca.groupby(["School Name", "Course Name"])["Enrollment Count"]
    .mean().reset_index()
    .rename(columns={"Enrollment Count": "school_course_avg"})
)
school_season_avg = (
    ca.groupby(["School Name", "season"])["Enrollment Count"]
    .mean().reset_index()
    .rename(columns={"Enrollment Count": "school_season_avg"})
)
course_avg = (
    ca.groupby("Course Name")["Enrollment Count"]
    .mean().reset_index()
    .rename(columns={"Enrollment Count": "course_avg"})
)

ca = (
    ca
    .merge(school_course_avg, on=["School Name", "Course Name"], how="left")
    .merge(school_season_avg, on=["School Name", "season"],       how="left")
    .merge(course_avg,        on="Course Name",                   how="left")
)

le_risk   = LabelEncoder().fit(["Low", "Medium", "High"])
le_season = LabelEncoder().fit(["Fall", "Spring", "Unknown"])
ca["risk_encoded"]   = le_risk.transform(ca["risk_final"].fillna("Medium"))
ca["season_encoded"] = le_season.transform(ca["season"].fillna("Unknown"))
ca["is_lea_int"]     = ca["is_lea_partner"].astype(int)
ca["is_fall"]        = (ca["season"] == "Fall").astype(int)

FEATURES = [
    "school_course_avg",
    "school_season_avg",
    "course_avg",
    "avg_participating_per_ca",
    "school_enrollment_cv",
    "ca_drop_rate",
    "survival_rate",
    "risk_encoded",
    "is_fall",
    "is_lea_int",
]

# Training set: non-dropped CAs only, enrollment cap at 50
train = ca[~ca["End Status"].isin(TRUE_DROPS)].copy()
ENROLLMENT_CAP = 50
train = train[train["Enrollment Count"] <= ENROLLMENT_CAP]
X     = train[FEATURES].fillna(train[FEATURES].median())
y     = train["Enrollment Count"]

print(f"Training set: {len(train)} CAs")
print(f"Enrollment distribution:\n{y.describe().round(1)}")

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = Ridge(alpha=1.0)
gbr   = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=5, random_state=42,
)

ridge_mae = -cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
gbr_mae   = -cross_val_score(gbr,   X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
ridge_r2  =  cross_val_score(ridge, X, y, cv=kf, scoring="r2").mean()
gbr_r2    =  cross_val_score(gbr,   X, y, cv=kf, scoring="r2").mean()

print(f"\nRidge  CV MAE: {ridge_mae:.2f}  R²: {ridge_r2:.3f}")
print(f"GBR    CV MAE: {gbr_mae:.2f}  R²: {gbr_r2:.3f}")

best_model = ridge if ridge_mae <= gbr_mae else gbr
best_name  = "Ridge" if ridge_mae <= gbr_mae else "GradientBoosting"
best_model.fit(X, y)
print(f"Selected: {best_name}")

train["predicted"] = best_model.predict(X).clip(min=0)
for tier in ["High", "Medium", "Low"]:
    subset = train[train["risk_final"] == tier]
    if len(subset):
        mae = mean_absolute_error(subset["Enrollment Count"], subset["predicted"])
        print(f"  {tier:8s} (n={len(subset):4d})  MAE={mae:.2f}")

X_all = ca[FEATURES].fillna(train[FEATURES].median())
ca["predicted_enrollment"] = best_model.predict(X_all).clip(min=0)

with open("school_enrollment_model.pkl", "wb") as f:
    pickle.dump({
        "model":         best_model,
        "model_name":    best_name,
        "features":      FEATURES,
        "train_medians": train[FEATURES].median().to_dict(),
    }, f)

out_cols_2 = [
    "Course Allocation ID", "School Name", "Course Name", "Term Name",
    "season", "risk_final", "End Status",
    "Enrollment Count", "predicted_enrollment",
]
ca[out_cols_2].to_csv("ca_enrollment_predictions.csv", index=False)
print("\nSaved → school_enrollment_model.pkl")
print("Saved → ca_enrollment_predictions.csv")


# =============================================================================
# MODULE 3 — Dynamic CA Threshold
# =============================================================================
print("\n" + "="*60)
print("MODULE 3 — Dynamic CA Threshold")
print("="*60)

preds = pd.read_csv("ca_enrollment_predictions.csv")
preds = preds.merge(
    risk[["School Name", "survival_rate", "risk_final", "ca_drop_rate", "avg_participating_per_ca"]],
    on="School Name", how="left", suffixes=("", "_r"),
)
preds["survival_rate"] = preds["survival_rate"].fillna(risk["survival_rate"].median())
preds["risk_final"]    = preds["risk_final"].fillna("Medium")

# Tier-specific offer conversion rates
ca_fresh = pd.read_csv("New_Course_Allocation.csv")
offered  = ca_fresh.merge(
    risk[["School Name", "risk_final"]], on="School Name", how="left"
)
offered = offered[offered["Starting Stage"] == "Offered"]

tier_conversion = (
    offered.groupby("risk_final")
    .apply(lambda x: x["End Status"].isin(["Active","Archived"]).mean())
    .to_dict()
)
GLOBAL_CONVERSION = offered["End Status"].isin(["Active","Archived"]).mean()
tier_conversion["Unknown"] = GLOBAL_CONVERSION

print("Tier-specific offer conversion rates:")
for tier, rate in sorted(tier_conversion.items()):
    print(f"  {tier:8s}: {rate:.1%}")

# Course Seat Goals with improved fallback
course_goals = (
    course[["Course Name", "Term Name", "Seat Goal", "Seat Capacity"]]
    .drop_duplicates(["Course Name", "Term Name"])
)
course_median_goal = (
    course_goals.groupby("Course Name")["Seat Goal"]
    .median().reset_index()
    .rename(columns={"Seat Goal": "median_seat_goal"})
)
all_ct = preds[["Course Name", "Term Name"]].drop_duplicates()
all_ct = all_ct.merge(course_goals, on=["Course Name", "Term Name"], how="left")
all_ct = all_ct.merge(course_median_goal, on="Course Name", how="left")

nan_goal = all_ct["Seat Goal"].isna()
all_ct.loc[nan_goal, "Seat Goal"] = all_ct.loc[nan_goal, "median_seat_goal"]

global_median = course_goals["Seat Goal"].median()
still_missing = all_ct["Seat Goal"].isna()
all_ct.loc[still_missing, "Seat Goal"]     = global_median
all_ct.loc[still_missing, "Seat Capacity"] = global_median / 1.2
course_goals = all_ct.drop(columns=["median_seat_goal"])

if nan_goal.sum():
    print(f"{nan_goal.sum()} course-term rows: Seat Goal filled from course median")
if still_missing.sum():
    print(f"{still_missing.sum()} course-term rows: Seat Goal filled from global median ({global_median:.0f})")

PROGRAM_AVG = preds.loc[
    ~preds["End Status"].isin(TRUE_DROPS), "Enrollment Count"
].mean()
print(f"Program-level avg enrollment per CA: {PROGRAM_AVG:.1f}")

def compute_threshold(group, seat_goal):
    group = group.copy()
    group["expected"] = group["predicted_enrollment"] * group["survival_rate"]

    active         = group[~group["End Status"].isin(TRUE_DROPS)]
    n_active       = len(active)
    expected_total = active["expected"].sum()
    actual_total   = active["Enrollment Count"].sum()
    avg_expected   = active["expected"].mean() if n_active > 0 else PROGRAM_AVG * 0.668

    accepted_threshold = np.ceil(seat_goal / avg_expected) if avg_expected > 0 else np.nan
    old_threshold      = np.ceil(seat_goal / PROGRAM_AVG)

    if n_active > 0:
        active = active.copy()
        active["offer_conversion"] = active["risk_final"].map(tier_conversion).fillna(GLOBAL_CONVERSION)
        weighted_conversion = active["offer_conversion"].mean()
    else:
        weighted_conversion = GLOBAL_CONVERSION

    offer_threshold = np.ceil(accepted_threshold / weighted_conversion) if not np.isnan(accepted_threshold) else np.nan

    return pd.Series({
        "n_active_ca":           n_active,
        "n_dropped_ca":          group["End Status"].isin(TRUE_DROPS).sum(),
        "expected_enrollment":   round(expected_total, 1),
        "actual_enrollment":     actual_total,
        "gap_to_seat_goal":      round(seat_goal - expected_total, 1),
        "accepted_ca_threshold": int(accepted_threshold) if not np.isnan(accepted_threshold) else np.nan,
        "offer_ca_threshold":    int(offer_threshold)    if not np.isnan(offer_threshold)    else np.nan,
        "old_ca_threshold":      int(old_threshold)      if not np.isnan(old_threshold)      else np.nan,
        "threshold_delta":       int(accepted_threshold - old_threshold) if (not np.isnan(accepted_threshold) and not np.isnan(old_threshold)) else np.nan,
        "offer_vs_old_delta":    int(offer_threshold - old_threshold)    if (not np.isnan(offer_threshold)    and not np.isnan(old_threshold)) else np.nan,
    })

results = []
for (course_name, term_name), group in preds.groupby(["Course Name", "Term Name"]):
    meta = course_goals[
        (course_goals["Course Name"] == course_name) &
        (course_goals["Term Name"]   == term_name)
    ]
    if meta.empty:
        continue
    seat_goal = meta["Seat Goal"].values[0]
    seat_cap  = meta["Seat Capacity"].values[0]
    metrics   = compute_threshold(group, seat_goal)
    results.append({
        "Course Name":  course_name,
        "Term Name":    term_name,
        "Seat Goal":    seat_goal,
        "Seat Cap":     seat_cap,
        **metrics.to_dict()
    })

thresholds = pd.DataFrame(results)
thresholds["fulfillment_rate"] = (
    thresholds["expected_enrollment"] / thresholds["Seat Goal"]
).round(3)
thresholds["actual_fill_rate"] = (
    thresholds["actual_enrollment"] / thresholds["Seat Goal"]
).round(3)
thresholds["fulfillment_status"] = thresholds["fulfillment_rate"].apply(
    lambda r: "Over" if r > 1.15 else ("Under" if r < 0.85 else "On Track")
)

print(f"\nFulfillment status (active courses):")
print(thresholds[thresholds["n_active_ca"] > 0]["fulfillment_status"].value_counts())
print(f"\nAccepted threshold delta — mean: {thresholds['threshold_delta'].mean():.1f}, max: {thresholds['threshold_delta'].max()}")
print(f"Offer threshold delta    — mean: {thresholds['offer_vs_old_delta'].mean():.1f}, max: {thresholds['offer_vs_old_delta'].max()}")

# SST Intervention Flags
active_preds = preds[~preds["End Status"].isin(TRUE_DROPS)].copy()
active_preds["expected"] = (
    active_preds["predicted_enrollment"] * active_preds["survival_rate"]
)
active_preds["flag_high_risk"] = active_preds["risk_final"] == "High"
active_preds["flag_squatting"] = (
    (active_preds["predicted_enrollment"] < 5) &
    (active_preds["risk_final"] != "Low")
)
active_preds["needs_intervention"] = (
    active_preds["flag_high_risk"] | active_preds["flag_squatting"]
)

flags = (
    active_preds[active_preds["needs_intervention"]]
    [[
        "School Name", "Course Name", "Term Name", "risk_final",
        "predicted_enrollment", "expected",
        "flag_high_risk", "flag_squatting",
    ]]
    .sort_values(["Term Name", "expected"])
    .reset_index(drop=True)
)

print(f"\nSST intervention flags: {len(flags)}")
print(f"  High-risk: {flags['flag_high_risk'].sum()}")
print(f"  Low Utilization: {flags['flag_squatting'].sum()}")

thresholds.to_csv("course_thresholds.csv", index=False)
flags.to_csv("sst_intervention_flags.csv", index=False)
print("\nSaved → course_thresholds.csv")
print("Saved → sst_intervention_flags.csv")

print("\nPipeline complete.")
print("Output files:")
print("  school_risk_profile.csv")
print("  ca_enrollment_predictions.csv")
print("  school_enrollment_model.pkl")
print("  course_thresholds.csv")
print("  sst_intervention_flags.csv")