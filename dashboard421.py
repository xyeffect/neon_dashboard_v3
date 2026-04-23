"""
SST Enrollment Forecasting Dashboard
Run with: streamlit run sst_dashboard.py

Requires:
  - course_thresholds.csv
  - school_risk_profile.csv
  - sst_intervention_flags.csv
  - ca_enrollment_predictions.csv
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NEON SST Enrollment Dashboard", layout="wide")

@st.cache_data
def load_data():
    thresholds = pd.read_csv("course_thresholds.csv")
    risk       = pd.read_csv("school_risk_profile.csv")
    flags      = pd.read_csv("sst_intervention_flags.csv")
    preds      = pd.read_csv("ca_enrollment_predictions.csv")
    return thresholds, risk, flags, preds

thresholds, risk, flags, preds = load_data()

TRUE_DROPS = ["Admissions Drop", "Dropped"]
LATE_ENROLLER_THRESHOLD = 0.30  # 30%+ students enroll within 30 days of start
LATE_DROPPER_THRESHOLD  = 0.40  # 40%+ drops happen within 30 days of start

# Current actual enrollment per course-term
current_actual = (
    preds[~preds["End Status"].isin(TRUE_DROPS)]
    .groupby(["Course Name", "Term Name"])["Enrollment Count"]
    .sum().reset_index()
    .rename(columns={"Enrollment Count": "current_actual_enrollment"})
)
thresholds = thresholds.merge(current_actual, on=["Course Name", "Term Name"], how="left")
thresholds["current_actual_enrollment"] = thresholds["current_actual_enrollment"].fillna(0)

# Risk concentration per course-term
risk_conc = (
    preds[~preds["End Status"].isin(TRUE_DROPS)]
    .merge(risk[["School Name","risk_final"]], on="School Name", how="left", suffixes=("","_r"))
    .groupby(["Course Name","Term Name"])
    .apply(lambda x: pd.Series({
        "pct_high_risk": (x["risk_final_r"] == "High").mean(),
        "n_high_risk":   (x["risk_final_r"] == "High").sum(),
    }))
    .reset_index()
)
thresholds = thresholds.merge(risk_conc, on=["Course Name","Term Name"], how="left")
thresholds["pct_high_risk"] = thresholds["pct_high_risk"].fillna(0)

# Identify current vs historical terms
term_active_counts = thresholds.groupby("Term Name")["n_active_ca"].sum()
CURRENT_TERMS = set(term_active_counts[term_active_counts > 0].index)
all_terms     = sorted(thresholds["Term Name"].dropna().unique(), reverse=True)

# Session state for selected course
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None

# Sidebar
st.sidebar.title("Filters")
selected_term = st.sidebar.selectbox("Term", all_terms, index=0)
is_historical = selected_term not in CURRENT_TERMS

if is_historical:
    st.sidebar.info("Historical term — showing actual enrollment results.")
    active_only = False
else:
    active_only = st.sidebar.checkbox("Active courses only (n_active_ca > 0)", value=True)

# Filter thresholds
t = thresholds[thresholds["Term Name"] == selected_term].copy()
if active_only:
    t = t[t["n_active_ca"] > 0]

if is_historical:
    t["display_enrollment"] = t["actual_enrollment"].fillna(0)
    t["display_label"]      = "Actual Enrollment"
else:
    t["display_enrollment"] = t["expected_enrollment"]
    t["display_label"]      = "Expected Enrollment"

t["display_fill_rate"] = (t["display_enrollment"] / t["Seat Goal"]).round(3)
t["display_status"]    = t["display_fill_rate"].apply(
    lambda r: "Over" if r > 1.15 else ("Under" if r < 0.85 else "On Track")
)

# Sort within each status group descending by fill rate
status_order = {"Over": 0, "On Track": 1, "Under": 2}
t["status_order"] = t["display_status"].map(status_order)
t = t.sort_values(["status_order", "display_fill_rate"], ascending=[True, False])

# Header
st.title("NEON SST Enrollment Dashboard")
st.caption(f"Term: {selected_term}  |  {len(t)} courses  |  Forecasting pipeline output")

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
on_track      = (t["display_status"] == "On Track").sum()
under         = (t["display_status"] == "Under").sum()
over          = (t["display_status"] == "Over").sum()
total_display = t["display_enrollment"].sum()
total_goal    = t["Seat Goal"].sum()
n_flags       = len(flags[flags["Term Name"] == selected_term])

enroll_label = "Actual Enrollment" if is_historical else "Expected Enrollment"
k1.metric("Courses On Track", f"{on_track}")
k2.metric("Courses Under",    f"{under}",  delta=f"-{under}", delta_color="inverse")
k3.metric("Courses Over",     f"{over}")
k4.metric(enroll_label,       f"{int(total_display):,}",
          delta=f"{int(total_display - total_goal):+,} vs goal")
k5.metric("Schools Flagged",  f"{n_flags}",
          delta="need SST action" if n_flags else None,
          delta_color="inverse")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Course Fulfillment",
    "School Risk Tiers",
    "School Watchlist",
    "Threshold: Old vs New",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Course Fulfillment — {selected_term}")
    if is_historical:
        st.caption("Showing actual enrollment results.")
    else:
        st.caption(
            "Expected enrollment = sum of (predicted enrollment x survival rate) across active CAs. "
            "Currently participating = students enrolled to date. "
            "Bars sorted by fulfillment rate within each status group. "
            "Target band: 85-115% of Seat Goal."
        )

    if t.empty:
        st.info("No courses found for this term.")
    else:
        COLOR_MAP = {"On Track": "#1D9E75", "Under": "#E24B4A", "Over": "#BA7517"}

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Seat Goal",
            x=t["Course Name"],
            y=t["Seat Goal"],
            marker_color="rgba(136,135,128,0.2)",
            marker_line_color="rgba(136,135,128,0.5)",
            marker_line_width=1,
        ))
        fig.add_trace(go.Bar(
            name=t["display_label"].iloc[0],
            x=t["Course Name"],
            y=t["display_enrollment"],
            marker_color=[COLOR_MAP[s] for s in t["display_status"]],
            text=[f"{r:.0%}" for r in t["display_fill_rate"]],
            textposition="outside",
            customdata=t[[
                "Seat Goal", "display_enrollment",
                "current_actual_enrollment", "n_active_ca",
                "display_status", "pct_high_risk", "offer_ca_threshold"
            ]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Expected: %{customdata[1]:.0f}<br>"
                "Currently participating: %{customdata[2]:.0f}<br>"
                "Seat Goal: %{customdata[0]:.0f}<br>"
                "Active CAs: %{customdata[3]}<br>"
                "High-risk CAs: %{customdata[5]:.0%}<br>"
                "CAs to offer: %{customdata[6]}<br>"
                "Status: %{customdata[4]}<extra></extra>"
            ),
        ))
        if not is_historical:
            fig.add_trace(go.Scatter(
                name="Currently Participating",
                x=t["Course Name"],
                y=t["current_actual_enrollment"],
                mode="markers",
                marker=dict(symbol="diamond", size=8, color="#042C53"),
                hovertemplate="<b>%{x}</b><br>Currently participating: %{y:.0f}<extra></extra>",
            ))
        fig.update_layout(
            barmode="overlay", xaxis_tickangle=-40, height=480,
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=40, b=10), yaxis_title="Students",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )

        clicked = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", key="fulfillment_chart"
        )
        if clicked and clicked.get("selection") and clicked["selection"].get("points"):
            pt = clicked["selection"]["points"][0]
            st.session_state.selected_course = pt.get("x")

        course_options = ["All courses"] + sorted(t["Course Name"].unique().tolist())
        default_idx = 0
        if st.session_state.selected_course in course_options:
            default_idx = course_options.index(st.session_state.selected_course)

        selected_course = st.selectbox(
            "Select a course to see school details",
            course_options,
            index=default_idx,
            key="course_selector"
        )
        if selected_course != "All courses":
            st.session_state.selected_course = selected_course
        else:
            st.session_state.selected_course = None

        col_a, col_b = st.columns([1, 2])
        with col_a:
            status_counts = t["display_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_pie = px.pie(
                status_counts, values="Count", names="Status",
                color="Status", color_discrete_map=COLOR_MAP, hole=0.5,
            )
            fig_pie.update_layout(
                height=260, margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            show_col   = "actual_enrollment" if is_historical else "expected_enrollment"
            show_label = "Actual" if is_historical else "Expected"
            t_display  = t.copy()
            if st.session_state.selected_course:
                t_display = t_display[t_display["Course Name"] == st.session_state.selected_course]

            display_df = t_display[[
                "Course Name", "Seat Goal", show_col,
                "current_actual_enrollment", "pct_high_risk",
                "offer_ca_threshold", "n_active_ca",
                "display_fill_rate", "display_status"
            ]].copy()
            display_df.columns = [
                "Course Name", "Seat Goal", show_label,
                "Currently Participating", "% High Risk CAs",
                "CAs to Offer", "Active CAs",
                "Fill Rate", "Status"
            ]
            st.dataframe(
                display_df.style
                .format({
                    show_label:               "{:.0f}",
                    "Currently Participating": "{:.0f}",
                    "% High Risk CAs":         "{:.0%}",
                    "Fill Rate":               "{:.1%}",
                })
                .applymap(
                    lambda v: "color: #E24B4A" if v == "Under"
                         else "color: #1D9E75" if v == "On Track"
                         else "color: #BA7517",
                    subset=["Status"],
                ),
                use_container_width=True, height=240,
            )

        if st.session_state.selected_course:
            st.subheader(f"Active schools — {st.session_state.selected_course}")
            course_preds = preds[
                (preds["Course Name"] == st.session_state.selected_course) &
                (preds["Term Name"]   == selected_term) &
                (~preds["End Status"].isin(TRUE_DROPS))
            ].merge(
                risk[["School Name","risk_final","survival_rate",
                      "ca_drop_rate","ca_late_drop_rate",
                      "late_enroll_rate","avg_participating_per_ca"]],
                on="School Name", how="left", suffixes=("_pred","")
            )
            if "risk_final_pred" in course_preds.columns:
                course_preds = course_preds.drop(columns=["risk_final_pred"])
            course_preds["expected"] = (
                course_preds["predicted_enrollment"] * course_preds["survival_rate"]
            )
            course_preds["late_enroller"] = (
                course_preds["late_enroll_rate"] >= LATE_ENROLLER_THRESHOLD
            )
            course_preds["late_dropper"] = (
                course_preds["ca_late_drop_rate"] >= LATE_DROPPER_THRESHOLD
            )
            school_table = course_preds[[
                "School Name", "risk_final", "Enrollment Count",
                "predicted_enrollment", "expected",
                "ca_drop_rate", "late_enroller", "late_dropper"
            ]].rename(columns={
                "risk_final":           "Risk Tier",
                "Enrollment Count":     "Currently Participating",
                "predicted_enrollment": "Predicted",
                "expected":             "Expected (x survival)",
                "ca_drop_rate":         "CA Drop Rate",
                "late_enroller":        "Late Enroller",
                "late_dropper":         "Late Dropper",
            }).sort_values("Expected (x survival)")

            st.dataframe(
                school_table.style
                .format({
                    "Currently Participating": "{:.0f}",
                    "Predicted":               "{:.1f}",
                    "Expected (x survival)":   "{:.1f}",
                    "CA Drop Rate":            "{:.1%}",
                })
                .applymap(
                    lambda v: "color: #E24B4A" if v == "High"
                         else "color: #1D9E75" if v == "Low"
                         else "color: #BA7517",
                    subset=["Risk Tier"],
                ),
                use_container_width=True,
            )

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("School Risk Tier Distribution")

    tier_order  = ["High", "Medium", "Low"]
    tier_colors = {"High": "#E24B4A", "Medium": "#BA7517", "Low": "#1D9E75"}

    col1, col2 = st.columns(2)
    with col1:
        tier_counts = risk["risk_final"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_bar = px.bar(
            tier_counts.set_index("Tier").reindex(tier_order).reset_index(),
            x="Tier", y="Count", color="Tier",
            color_discrete_map=tier_colors, text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=320, showlegend=False,
            yaxis_title="Number of Schools",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_box = px.box(
            risk, x="risk_final", y="ca_drop_rate",
            color="risk_final", color_discrete_map=tier_colors,
            category_orders={"risk_final": tier_order},
            labels={"risk_final": "Risk Tier", "ca_drop_rate": "CA Drop Rate"},
        )
        fig_box.update_layout(
            height=320, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Tier Attrition, Survival and Offer Conversion Rates")
    col_t, col_o = st.columns(2)

    with col_t:
        tier_summary = (
            risk.groupby("risk_final")
            .agg(
                Schools                  = ("School Name",             "count"),
                Avg_Drop_Rate            = ("ca_drop_rate",            "mean"),
                Avg_Survival             = ("survival_rate",           "mean"),
                Avg_Participating_per_CA = ("avg_participating_per_ca","mean"),
            )
            .reindex(tier_order)
            .reset_index()
            .rename(columns={"risk_final": "Tier"})
        )
        st.dataframe(
            tier_summary.style
            .format({
                "Avg_Drop_Rate":            "{:.1%}",
                "Avg_Survival":             "{:.1%}",
                "Avg_Participating_per_CA": "{:.1f}",
            })
            .applymap(
                lambda v: "color: #E24B4A" if v == "High"
                     else "color: #1D9E75" if v == "Low"
                     else "color: #BA7517",
                subset=["Tier"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_o:
        st.caption(
            "Offer conversion rate = share of Offered CAs that ultimately "
            "delivered. Used to calculate how many CAs NEON needs to offer "
            "to hit Seat Goal."
        )
        conversion_df = pd.DataFrame([
            {"Tier": "High",   "Offer Conversion Rate": 0.301, "Meaning": "1 in 3 offered CAs deliver"},
            {"Tier": "Medium", "Offer Conversion Rate": 0.677, "Meaning": "2 in 3 offered CAs deliver"},
            {"Tier": "Low",    "Offer Conversion Rate": 0.827, "Meaning": "4 in 5 offered CAs deliver"},
        ])
        st.dataframe(
            conversion_df.style
            .format({"Offer Conversion Rate": "{:.1%}"})
            .applymap(
                lambda v: "color: #E24B4A" if v == "High"
                     else "color: #1D9E75" if v == "Low"
                     else "color: #BA7517",
                subset=["Tier"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("CA Drop Rate vs. Avg Participating per CA")
    fig_scatter = px.scatter(
        risk, x="avg_participating_per_ca", y="ca_drop_rate",
        color="risk_final", color_discrete_map=tier_colors,
        hover_name="School Name",
        hover_data={"school_enrollment_cv": ":.2f", "ca_late_drop_rate": ":.2f"},
        labels={
            "avg_participating_per_ca": "Avg Participating per CA",
            "ca_drop_rate":             "CA Drop Rate",
            "risk_final":               "Risk Tier",
        },
        opacity=0.7,
        category_orders={"risk_final": tier_order},
    )
    fig_scatter.update_layout(
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 3 — School Watchlist ──────────────────────────────────────────────────
with tab3:
    st.subheader("School Watchlist")
    st.caption(
        "High-risk schools with active CAs, or schools showing low utilization behavior. "
        "Late Enroller = school where 30%+ of students historically enroll within 30 days of course start, "
        "based on actual student enrollment dates. "
        "Late Dropper = school where 40%+ of CA drops happen within 30 days of course start. "
        "Sorted by expected enrollment — lowest first indicates highest urgency."
    )

    term_flags = flags[flags["Term Name"] == selected_term].copy()

    if term_flags.empty:
        st.success(f"No schools on watchlist for {selected_term}.")
    else:
        term_flags = term_flags.merge(
            risk[["School Name","late_enroll_rate","ca_late_drop_rate","ca_drop_rate"]],
            on="School Name", how="left"
        )
        term_flags["late_enroller"] = (
            term_flags["late_enroll_rate"] >= LATE_ENROLLER_THRESHOLD
        )
        term_flags["late_dropper"] = (
            term_flags["ca_late_drop_rate"] >= LATE_DROPPER_THRESHOLD
        )

        def urgency_label(row):
            if row["flag_squatting"]:
                return "Low Utilization"
            if row["expected"] < 2:
                return "Critical"
            return "High Risk"

        term_flags["Urgency"] = term_flags.apply(urgency_label, axis=1)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Flags",        len(term_flags))
        c2.metric("Critical (< 2 exp)", (term_flags["expected"] < 2).sum())
        c3.metric("Low Utilization",    term_flags["flag_squatting"].sum())
        c4.metric("Late Enrollers",     term_flags["late_enroller"].sum())
        c5.metric("Late Droppers",      term_flags["late_dropper"].sum())

        if not t.empty:
            high_conc = t[t["pct_high_risk"] >= 0.5]
            if len(high_conc) > 0:
                st.warning(
                    f"{len(high_conc)} course(s) have 50%+ of active CAs from High-risk schools — "
                    f"these courses are structurally vulnerable even if current participation looks acceptable. "
                    f"Courses: {', '.join(high_conc['Course Name'].tolist())}"
                )

        st.dataframe(
            term_flags[[
                "Urgency", "School Name", "Course Name",
                "predicted_enrollment", "expected",
                "flag_high_risk", "flag_squatting",
                "late_enroller", "late_dropper", "ca_drop_rate",
            ]].rename(columns={
                "predicted_enrollment": "Predicted",
                "expected":             "Expected (x survival)",
                "flag_high_risk":       "High Risk",
                "flag_squatting":       "Low Utilization",
                "late_enroller":        "Late Enroller",
                "late_dropper":         "Late Dropper",
                "ca_drop_rate":         "CA Drop Rate",
            })
            .sort_values("Expected (x survival)")
            .style.format({
                "Predicted":            "{:.1f}",
                "Expected (x survival)":"{:.1f}",
                "CA Drop Rate":         "{:.1%}",
            }),
            use_container_width=True,
            height=420,
        )

        csv = term_flags.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download watchlist as CSV",
            data=csv,
            file_name=f"watchlist_{selected_term.replace(' ', '_')}.csv",
            mime="text/csv",
        )

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("CA Threshold: Old vs Accepted vs Offer")
    st.caption(
        "Old: Seat Goal divided by program-level avg (17.5 students per CA) — no risk adjustment.  "
        "Accepted: how many accepted CAs are needed, accounting for school-level drop risk.  "
        "Offer: how many CAs NEON needs to offer, accounting for both drop risk "
        "and the probability that offered CAs will be accepted and delivered."
    )

    t_thresh = thresholds[thresholds["Term Name"] == selected_term].copy()
    if not is_historical and active_only:
        t_thresh = t_thresh[t_thresh["n_active_ca"] > 0]

    if t_thresh.empty:
        st.info("No courses found for this term.")
    else:
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Bar(
            name="Old Threshold",
            x=t_thresh["Course Name"],
            y=t_thresh["old_ca_threshold"],
            marker_color="rgba(136,135,128,0.5)",
        ))
        fig_thresh.add_trace(go.Bar(
            name="Accepted CA Threshold",
            x=t_thresh["Course Name"],
            y=t_thresh["accepted_ca_threshold"],
            marker_color="#534AB7",
        ))
        fig_thresh.add_trace(go.Bar(
            name="Offer CA Threshold",
            x=t_thresh["Course Name"],
            y=t_thresh["offer_ca_threshold"],
            marker_color="#D85A30",
        ))
        fig_thresh.update_layout(
            barmode="group", height=420, xaxis_tickangle=-40,
            yaxis_title="Number of CAs",
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Accepted threshold vs Old**")
            st.caption("Additional accepted CAs needed vs old model")
            t_s1 = t_thresh.sort_values("threshold_delta", ascending=False)
            fig_d1 = px.bar(
                t_s1, x="Course Name", y="threshold_delta",
                color="threshold_delta",
                color_continuous_scale=["#1D9E75", "#BA7517", "#E24B4A"],
                text="threshold_delta",
            )
            fig_d1.update_traces(textposition="outside")
            fig_d1.update_layout(
                height=320, xaxis_tickangle=-40,
                yaxis_title="Delta (CAs)",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_d1, use_container_width=True)

        with col_d2:
            st.markdown("**Offer threshold vs Old**")
            st.caption("Additional CAs NEON needs to offer vs old model")
            t_s2 = t_thresh.sort_values("offer_vs_old_delta", ascending=False)
            fig_d2 = px.bar(
                t_s2, x="Course Name", y="offer_vs_old_delta",
                color="offer_vs_old_delta",
                color_continuous_scale=["#1D9E75", "#BA7517", "#E24B4A"],
                text="offer_vs_old_delta",
            )
            fig_d2.update_traces(textposition="outside")
            fig_d2.update_layout(
                height=320, xaxis_tickangle=-40,
                yaxis_title="Delta (CAs)",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_d2, use_container_width=True)

        st.subheader("Summary")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Accepted delta mean", f"+{t_thresh['threshold_delta'].mean():.1f} CAs")
        col_s2.metric("Accepted delta max",  f"+{t_thresh['threshold_delta'].max():.0f} CAs")
        col_s3.metric("Offer delta mean",    f"+{t_thresh['offer_vs_old_delta'].mean():.1f} CAs")
        col_s4.metric("Offer delta max",     f"+{t_thresh['offer_vs_old_delta'].max():.0f} CAs")