"""Customer segmentation model and manager-facing business helpers."""

import os
from pathlib import Path

# Set this before importing NumPy/scikit-learn to avoid an OpenMP warning on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("Mall_Customers.csv")
FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]
N_CLUSTERS = 5


def load_data() -> pd.DataFrame:
    """Load the dataset and fail early if the required fields are absent."""
    data = pd.read_csv(DATA_PATH)
    required_columns = {"CustomerID", "Gender", "Age", *FEATURE_COLUMNS}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Income and spending score cannot contain missing values.")
    return data


def _income_band(value: float, overall_mean: float) -> str:
    if value >= overall_mean * 1.10:
        return "high"
    if value <= overall_mean * 0.90:
        return "low"
    return "average"


def _spending_band(value: float, overall_mean: float) -> str:
    if value >= overall_mean * 1.10:
        return "high"
    if value <= overall_mean * 0.90:
        return "low"
    return "average"


PROFILE_LIBRARY = {
    ("high", "high"): {
        "name": "Premium Customers",
        "value": "Very high",
        "opportunity": "Retain and deepen loyalty",
        "description": "Above-average income and a high spending score.",
        "action": "Prioritise VIP service, personalised premium recommendations, and loyalty rewards.",
        "why": "They combine purchasing capacity with strong current engagement.",
    },
    ("high", "low"): {
        "name": "Potential Premium Customers",
        "value": "High potential",
        "opportunity": "Increase engagement",
        "description": "Above-average income but a low spending score.",
        "action": "Test a personalised offer or product recommendation, then track whether engagement improves.",
        "why": "They have purchasing capacity but are not currently highly engaged.",
    },
    ("low", "high"): {
        "name": "Value-Seeking Customers",
        "value": "High engagement",
        "opportunity": "Protect value perception",
        "description": "Below-average income and a high spending score.",
        "action": "Use bundles, loyalty rewards, and value-led offers instead of broad price increases.",
        "why": "Their high engagement suggests that value and relevant offers matter to them.",
    },
    ("low", "low"): {
        "name": "Occasional Customers",
        "value": "Lower current value",
        "opportunity": "Low-cost re-engagement",
        "description": "Below-average income and a low spending score.",
        "action": "Use low-cost, targeted promotions and measure whether they increase engagement.",
        "why": "The segment has lower purchasing capacity and lower current engagement.",
    },
    ("average", "average"): {
        "name": "Core Customers",
        "value": "Stable",
        "opportunity": "Maintain and grow",
        "description": "Income and spending are close to the customer-base average.",
        "action": "Maintain service quality, support loyalty, and test complementary product recommendations.",
        "why": "They form a stable middle segment with room for gradual growth.",
    },
}


def _fallback_profile(income_band: str, spending_band: str) -> dict:
    """Provide a sensible description if a centre falls between the five main patterns."""
    return {
        "name": f"{income_band.title()} Income / {spending_band.title()} Spending",
        "value": "To be assessed",
        "opportunity": "Test targeted actions",
        "description": f"{income_band.title()} income and {spending_band} spending relative to the dataset average.",
        "action": "Run a small, targeted offer and measure the change in engagement before scaling it.",
        "why": "This segment should be validated with transaction and campaign-response data.",
    }


def _build_model(data: pd.DataFrame):
    features = data[FEATURE_COLUMNS]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)
    return scaler, scaled_features, model, labels


jan = load_data()
scaler, X_scaled, kmeans, labels = _build_model(jan)
jan["Cluster"] = labels


def _build_segment_profiles() -> dict[int, dict]:
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    overall_income = jan[FEATURE_COLUMNS[0]].mean()
    overall_spending = jan[FEATURE_COLUMNS[1]].mean()
    profiles = {}

    for cluster_id, (income, spending) in enumerate(centers):
        income_band = _income_band(income, overall_income)
        spending_band = _spending_band(spending, overall_spending)
        profile = PROFILE_LIBRARY.get(
            (income_band, spending_band),
            _fallback_profile(income_band, spending_band),
        ).copy()
        profile.update(
            {
                "cluster": cluster_id,
                "income_band": income_band,
                "spending_band": spending_band,
                "center_income": round(float(income), 1),
                "center_spending": round(float(spending), 1),
            }
        )
        profiles[cluster_id] = profile
    return profiles


segment_profiles = _build_segment_profiles()
cluster_labels = {cluster_id: profile["name"] for cluster_id, profile in segment_profiles.items()}
jan["Segment"] = jan["Cluster"].map(cluster_labels)


def get_cluster_centers() -> pd.DataFrame:
    """Return cluster centers and business-friendly segment labels."""
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    result = pd.DataFrame(centers, columns=FEATURE_COLUMNS)
    result["Cluster"] = result.index
    result["Segment"] = result["Cluster"].map(cluster_labels)
    return result[["Cluster", "Segment", *FEATURE_COLUMNS]]


def get_model_metrics() -> dict:
    """Return evaluation metrics for model transparency, not manager decision-making."""
    return {
        "silhouette": round(float(silhouette_score(X_scaled, jan["Cluster"])), 3),
        "davies_bouldin": round(float(davies_bouldin_score(X_scaled, jan["Cluster"])), 3),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_scaled, jan["Cluster"])), 1),
    }


def get_k_evaluation() -> pd.DataFrame:
    """Compare plausible cluster counts using the same scaled feature set."""
    evaluations = []
    for cluster_count in range(2, min(10, len(jan) - 1) + 1):
        candidate = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        candidate_labels = candidate.fit_predict(X_scaled)
        evaluations.append(
            {
                "Clusters": cluster_count,
                "Silhouette score": round(float(silhouette_score(X_scaled, candidate_labels)), 3),
                "Davies-Bouldin score": round(float(davies_bouldin_score(X_scaled, candidate_labels)), 3),
            }
        )
    return pd.DataFrame(evaluations)


def get_dashboard_stats() -> dict:
    return {
        "customers": len(jan),
        "clusters": jan["Cluster"].nunique(),
        "average_income": round(float(jan[FEATURE_COLUMNS[0]].mean()), 1),
        "average_spending": round(float(jan[FEATURE_COLUMNS[1]].mean()), 1),
        **get_model_metrics(),
    }


def get_segment_summary() -> pd.DataFrame:
    """Aggregate only the fields needed for manager-facing segment comparisons."""
    summary = (
        jan.groupby(["Cluster", "Segment"], as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            Average_Income=(FEATURE_COLUMNS[0], "mean"),
            Average_Spending=(FEATURE_COLUMNS[1], "mean"),
        )
        .sort_values("Customers", ascending=False)
    )
    summary["Share_of_Customers"] = summary["Customers"] / len(jan) * 100
    return summary.round({"Average_Income": 1, "Average_Spending": 1, "Share_of_Customers": 1})


def _comparison_text(value: float, benchmark: float, measure: str, unit: str = "") -> str:
    difference = value - benchmark
    direction = "above" if difference >= 0 else "below"
    return f"{measure} is {abs(difference):.1f}{unit} {direction} the segment average ({benchmark:.1f}{unit})."


def build_manager_report(income: float, spending: float, cluster: int, customer_id: int | None = None) -> dict:
    """Translate a cluster assignment into a transparent, manager-facing action plan."""
    profile = segment_profiles[int(cluster)]
    segment_data = jan[jan["Cluster"] == cluster]
    segment_income = float(segment_data[FEATURE_COLUMNS[0]].mean())
    segment_spending = float(segment_data[FEATURE_COLUMNS[1]].mean())
    customer_label = f"Customer {customer_id}" if customer_id is not None else "New customer"

    return {
        "customer_label": customer_label,
        "cluster": int(cluster),
        "segment": profile["name"],
        "segment_description": profile["description"],
        "customer_value": profile["value"],
        "opportunity": profile["opportunity"],
        "recommended_action": profile["action"],
        "reason": profile["why"],
        "income_comparison": _comparison_text(income, segment_income, "Annual income", "k"),
        "spending_comparison": _comparison_text(spending, segment_spending, "Spending score"),
        "segment_customers": int(len(segment_data)),
        "segment_share": round(len(segment_data) / len(jan) * 100, 1),
        "segment_average_income": round(segment_income, 1),
        "segment_average_spending": round(segment_spending, 1),
    }


def get_customer_info(customer_id: int) -> dict | None:
    """Look up an existing customer and return a manager-ready report."""
    customer = jan.loc[jan["CustomerID"] == int(customer_id)]
    if customer.empty:
        return None
    row = customer.iloc[0]
    report = build_manager_report(
        income=float(row[FEATURE_COLUMNS[0]]),
        spending=float(row[FEATURE_COLUMNS[1]]),
        cluster=int(row["Cluster"]),
        customer_id=int(row["CustomerID"]),
    )
    report.update({"age": int(row["Age"]), "gender": row["Gender"]})
    return report


def predict_customer(income: float, spending: float) -> dict:
    """Assign a new customer to a segment and return the action plan."""
    new_customer = pd.DataFrame([[income, spending]], columns=FEATURE_COLUMNS)
    cluster = int(kmeans.predict(scaler.transform(new_customer))[0])
    return build_manager_report(float(income), float(spending), cluster)
