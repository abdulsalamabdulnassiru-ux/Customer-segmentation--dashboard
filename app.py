"""Streamlit manager dashboard for the customer segmentation model."""

import plotly.express as px
import streamlit as st

from customer_segmentation import (
    FEATURE_COLUMNS,
    get_cluster_centers,
    get_customer_info,
    get_dashboard_stats,
    get_k_evaluation,
    get_segment_summary,
    jan,
    predict_customer,
    segment_profiles,
)


st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

def display_manager_report(report: dict) -> None:
    """Render the selected customer's segment and recommended manager action."""
    st.header(f"Manager action plan: {report['customer_label']}")
    st.caption(f"Segment {report['cluster'] + 1}: {report['segment']}")

    first, second, third, fourth = st.columns(4)
    first.metric("Segment", report["segment"])
    second.metric("Customer value", report["customer_value"])
    third.metric("Segment size", f"{report['segment_customers']} customers")
    fourth.metric("Segment share", f"{report['segment_share']}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Why this customer is in this segment")
        st.write(report["segment_description"])
        st.markdown(f"- {report['income_comparison']}")
        st.markdown(f"- {report['spending_comparison']}")
    with right:
        st.subheader("Recommended manager action")
        st.info(report["recommended_action"])
        st.markdown(f"**Why this action:** {report['reason']}")
        st.markdown(f"**Opportunity:** {report['opportunity']}")

    with st.expander("Segment benchmarks"):
        st.write(
            f"Average income: **${report['segment_average_income']:.1f}k**  \\n+Average spending score: **{report['segment_average_spending']:.1f}**"
        )


st.title("Customer Segmentation Dashboard")
st.write("A manager-facing view of customer segments built with K-means clustering.")
st.caption("Use a customer ID or enter a new customer to receive a segment explanation and an action to test.")

stats = get_dashboard_stats()
metrics = st.columns(4)
metrics[0].metric("Customers analysed", stats["customers"])
metrics[1].metric("Customer segments", stats["clusters"])
metrics[2].metric("Average annual income", f"${stats['average_income']:.1f}k")
metrics[3].metric("Average spending score", f"{stats['average_spending']:.1f}")

st.sidebar.header("Customer lookup")
mode = st.sidebar.radio("Choose an option", ["Existing customer", "New customer"])

if mode == "Existing customer":
    with st.sidebar.form("existing_customer_form"):
        customer_id = st.number_input("Customer ID", min_value=1, max_value=int(jan["CustomerID"].max()), value=1, step=1)
        submitted = st.form_submit_button("Show action plan")
    if submitted:
        st.session_state["manager_report"] = get_customer_info(int(customer_id))
else:
    with st.sidebar.form("new_customer_form"):
        income = st.number_input("Annual income (k$)", min_value=0.0, max_value=250.0, value=60.0, step=1.0)
        spending = st.number_input("Spending score (1–100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
        submitted = st.form_submit_button("Predict segment and show action plan")
    if submitted:
        st.session_state["manager_report"] = predict_customer(income, spending)

report = st.session_state.get("manager_report")
if report:
    st.divider()
    display_manager_report(report)

st.divider()
st.header("Segment overview")
summary = get_segment_summary()

left, right = st.columns(2)
with left:
    segment_count_chart = px.bar(
        summary.sort_values("Customers"),
        x="Customers",
        y="Segment",
        orientation="h",
        text="Customers",
        title="Customers in each segment",
        color="Segment",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    segment_count_chart.update_layout(showlegend=False, yaxis_title=None)
    st.plotly_chart(segment_count_chart, use_container_width=True)

with right:
    scatter_data = jan.copy()
    scatter = px.scatter(
        scatter_data,
        x=FEATURE_COLUMNS[0],
        y=FEATURE_COLUMNS[1],
        color="Segment",
        hover_data=["CustomerID", "Age", "Gender"],
        title="Customer segments by income and spending score",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    if report:
        scatter.add_scatter(
            x=[report.get("segment_average_income")],
            y=[report.get("segment_average_spending")],
            mode="markers",
            marker={"size": 15, "color": "black", "symbol": "x"},
            name="Segment average",
        )
    st.plotly_chart(scatter, use_container_width=True)

st.subheader("What each segment means")
for cluster_id, profile in segment_profiles.items():
    with st.expander(f"Segment {cluster_id + 1}: {profile['name']}"):
        st.write(profile["description"])
        st.markdown(f"**Recommended action:** {profile['action']}")
        st.markdown(f"**Reason:** {profile['why']}")

display_summary = summary.rename(
    columns={
        "Average_Income": "Average income (k$)",
        "Average_Spending": "Average spending score",
        "Share_of_Customers": "Share of customers (%)",
    }
)
st.subheader("Segment benchmark table")
st.dataframe(display_summary.drop(columns="Cluster"), use_container_width=True, hide_index=True)

with st.expander("Model details"):
    model_metrics = st.columns(3)
    model_metrics[0].metric("Silhouette score", stats["silhouette"])
    model_metrics[1].metric("Davies-Bouldin score", stats["davies_bouldin"])
    model_metrics[2].metric("Calinski-Harabasz score", stats["calinski_harabasz"])
    st.dataframe(get_cluster_centers().round(1), use_container_width=True, hide_index=True)
    k_evaluation = get_k_evaluation()
    recommended_k = int(k_evaluation.loc[k_evaluation["Silhouette score"].idxmax(), "Clusters"])
    st.markdown(f"**Cluster-count check:** the highest silhouette score in the 2–10 comparison occurs at **{recommended_k} clusters**.")
    st.dataframe(k_evaluation, use_container_width=True, hide_index=True)
    st.caption("These metrics help assess cluster separation. They are provided for model transparency, not as manager KPIs.")

st.divider()
st.caption(
    "Important: the Mall Customers dataset contains income and a spending score, not actual sales, purchase frequency, or campaign results. "
    "The recommended actions are hypotheses to test with real transaction and campaign data."
)
