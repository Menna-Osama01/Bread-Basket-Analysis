import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Retail Transaction Analysis with Apriori")
st.subheader("", divider="blue")

# --- Upload CSV File ---
file = st.file_uploader("Upload Transaction CSV File", type=["csv"])

if file is not None:
    # Load uploaded CSV
    data = pd.read_csv(file)

    st.subheader("Raw Data Preview")
    st.subheader("", divider="blue")
    st.dataframe(data)

    # --- Data Cleaning Function ---
    def load_data(df):
        # Cleaning
        df['Item'] = df['Item'].astype(str).str.replace(';', ',')
        df['Item'] = df['Item'].str.replace('\u00A0', ' ')
        df['Item'] = df['Item'].astype(str).str.strip().str.lower()
        df = df[~df['Item'].isin(['none', 'nan', '', 'nan ', 'none ', 'none.'])]

        # Date formatting
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df = df.dropna(subset=['date_time'])

        df['date'] = df['date_time'].dt.date
        df['time'] = df['date_time'].dt.time

        df['month'] = df['date_time'].dt.month.replace({
            1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
            7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'
        })

        hour_in_num = (1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
        hour_in_obj = ('1-2','7-8','8-9','9-10','10-11','11-12','12-13',
                       '13-14','14-15','15-16','16-17','17-18','18-19',
                       '19-20','20-21','21-22','22-23','23-24')
        df['hour'] = df['date_time'].dt.hour.replace(hour_in_num, hour_in_obj)

        df['weekday'] = df['date_time'].dt.weekday.replace({
            0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',
            4:'Friday',5:'Saturday',6:'Sunday'
        })

        df.drop('date_time', axis=1, inplace=True)
        df = df.drop_duplicates(subset=['Transaction', 'Item', 'date'], keep='first')
        return df

    df = load_data(data)

    st.subheader("Preview of Cleaned Dataset")
    st.subheader("", divider="blue")
    st.dataframe(df)

    # --- VISUALIZATION 1: Most Frequent Items ---
    st.header("Interactive Visualizations")
    st.subheader("", divider="blue")
    
    top_items = df['Item'].value_counts().head(15).reset_index()
    top_items.columns = ['Item', 'Count']
    fig_items = px.bar(top_items, x='Count', y='Item', orientation='h',
                       title='Top 15 Purchased Items', text='Count', color='Count')
    st.plotly_chart(fig_items, use_container_width=True)

    # Purchases by Month
    month_counts = df['month'].value_counts().sort_index().reset_index()
    month_counts.columns = ['Month', 'Count']
    fig_month = px.bar(
        month_counts,
        x='Month',
        y='Count',
        title='Purchases by Month',
        text='Count',
        color='Count'
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Purchases by Weekday
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)
    fig_weekday = px.histogram(df, x='weekday', title='Purchases by Weekday',
                               color_discrete_sequence=['orange'])
    st.plotly_chart(fig_weekday, use_container_width=True)

    # --- Transaction List ---
    transactions = df.groupby("Transaction")['Item'].apply(list).reset_index(name='items_list')
    st.subheader("Transaction List")
    st.dataframe(transactions)

    # Transaction Encoder
    te = TransactionEncoder()
    te_array = te.fit(transactions['items_list']).transform(transactions['items_list'])
    basket = pd.DataFrame(te_array, columns=te.columns_)
    st.write(f"Basket matrix shape: {basket.shape}")

    # --- APRIORI ASSOCIATION RULES ---
    st.header("Apriori Association Rule Mining")
    min_support = st.slider("Minimum Support", 0.01, 0.1, 0.02, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)

    if st.button("Run Apriori"):
        freq_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        freq_itemsets = freq_itemsets.sort_values('support', ascending=False).reset_index(drop=True)
        st.subheader("Frequent Itemsets")
        st.dataframe(freq_itemsets)

        # Association rules
        rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_conf)
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.subheader("Generated Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # --- VISUALIZATION 4: Top Rules (confidence & lift) ---
        st.subheader("Support vs Lift (All Rules)")
        st.subheader("", divider="blue")
        # Support vs Lift
        fig_support_lift = px.scatter(
            rules,
            x="support",
            y="lift",
            size="confidence",
            color="confidence",
            hover_data=["antecedents_str", "consequents_str"],
            title="Support vs Lift (Bubble Size = Confidence)"
        )
        st.plotly_chart(fig_support_lift, use_container_width=True)

        # Confidence Heatmap
        #Heatmap: Top Item-to-Item Confidence
        st.subheader("Confidence Heatmap (Top Rules)")
        st.subheader("", divider="blue")
        heatmap_data = rules.sort_values("confidence", ascending=False).head(15)
        heatmap_pivot = heatmap_data.pivot(
            index="antecedents_str",
            columns="consequents_str",
            values="confidence"
        )
        fig_heatmap, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_pivot, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        ax.set_title("Confidence Heatmap (Top 15 Rules)")
        st.pyplot(fig_heatmap)

        # Top Rules by Lift
        st.subheader("Top Rules by Lift")
        st.subheader("", divider="blue")
        rules['rule'] = rules['antecedents_str'] + " → " + rules['consequents_str']
        top_lift_rules = rules.sort_values("lift", ascending=False).head(10)
        fig_lift = px.bar(
            top_lift_rules,
            x="lift",
            y="rule",
            orientation="h",
            title="Top 10 Rules by Lift",
            color="confidence"
        )
        st.plotly_chart(fig_lift, use_container_width=True)

        st.balloons()
