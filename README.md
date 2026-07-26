# Customer-segmentation--dashboard

An interactive Streamlit web application that uses K-means clustering to segment mall customers by annual income and spending score.

## Project overview

I built this project to turn customer segmentation analysis into a simple decision-support tool for non-technical managers. Instead of only presenting model outputs, the dashboard explains what each customer segment represents and suggests an action a manager can test.

The model groups customers into five segments based on:

- Annual Income
- Spending Score

Each segment is given a business-friendly name, such as Premium Customers, Potential Premium Customers, Core Customers, Value-Seeking Customers, or Occasional Customers.

## Features

- Customer lookup using Customer ID
- Segment prediction for new customers using income and spending score
- Manager action plan for each selected customer
- Clear explanation of why a customer belongs to a segment
- Customer comparison against their segment’s average income and spending score
- Recommended business actions for each segment
- Interactive customer segmentation scatter plot
- Segment sizes, benchmark table, and cluster-centre overview
- Model evaluation using Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score
- Comparison of possible cluster counts from 2 to 10

## Tools used

- Python
- Pandas
- Scikit-learn
- K-means clustering
- Plotly
- Streamlit

## Important note

The dataset contains annual income and spending scores, rather than real transaction history, revenue, or campaign results. Therefore, the manager recommendations in this dashboard are business hypotheses to test with real customer and sales data.
