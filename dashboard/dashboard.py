import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(
    page_title="Knoxicle - AccessGuru Analytics", 
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def local_css():
    st.markdown("""
        <style>
        /* Main background color */
        .stApp { background-color: #f4f7f6; }
        
        /* Metric Card Styling */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem;
            color: #1f77b4;
            font-weight: 700;
        }
        
        /* Card-like containers for charts */
        .plot-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Header styling */
        h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
        </style>
    """, unsafe_allow_html=True)

# --- Data Preprocessing ---
@st.cache_data
def load_and_preprocess_data():

    df = pd.read_csv("data/Access_to_Tech_Dataset.csv")

    # Filter out pages with unsuccessful scrapes (scrape_status)
    df_clean = df[df['scrape_status'] == 'scraped'].copy()
    
    # Map the messy values to clean, standardized names before you run your groupby aggregation.
    # From the prompt: domain_category – Type of site (health, education, government, news, tech, e-commerce)
    df_clean['domain_category'] = df_clean['domain_category'].str.strip()
    cleanup_map = {
        'E-commerce': 'e-commerce',
        'Ecommerce': 'e-commerce',
        'Educational Platforms': 'education', 
        'Government and Public Services': 'government',
        'Health and Wellness': 'health', 
        'News and Media': 'news',
        'Streaming Platforms': 'tech', 
        'Technology Science and Research': 'tech',
        'TechnologyScienceResearch': 'tech'
    }
    df_clean['domain_category'] = df_clean['domain_category'].replace(cleanup_map)
    return df_clean

# --- Plotting Functions ---
def plot_domain_rankings(df):
    counts = df['domain_category'].value_counts().reset_index()
    counts.columns = ['Domain', 'Violations']
    print(counts)
    fig = px.bar(counts, x='Violations', y='Domain', orientation='h', 
                 color='Violations', color_continuous_scale='Blues')
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def plot_top_violation_types(df):
    top_v = df['violation_name'].value_counts().head(10).reset_index()
    top_v.columns = ['Violation Name', 'Frequency']
    print(top_v)
    fig = px.bar(top_v, x='Frequency', y='Violation Name', orientation='h', color_discrete_sequence=['#1f77b4'])
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig

def plot_category_patterns(df):
    cat_counts = df['violation_category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    print(cat_counts)
    fig = px.pie(cat_counts, values='Count', names='Category', hole=0,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(l=50, r=20, t=30, b=20))
    return fig

def get_severe_pages(df):
    severe_pages = df.groupby(['web_URL']).agg(
        total_severity=('violation_score', 'sum'),
        error_count=('id', 'count')
    ).reset_index().sort_values('total_severity', ascending=False).head(10)

    severe_pages = severe_pages.reset_index(drop=True)
    severe_pages.index = severe_pages.index + 1
    severe_pages.index.name = "Rank"

    print(severe_pages)
    return severe_pages

def plot_heatmap(df):
    heatmap_data = pd.crosstab(df['domain_category'], df['violation_category'])
    fig = px.imshow(heatmap_data, text_auto=True, aspect="auto",
                    title="Heatmap: Domain vs. Violation Category",
                    labels=dict(x="Category", y="Domain", color="Count"),
                    color_continuous_scale='YlOrRd')
    return fig

def plot_invisible_barriers(df, selected_comparison_domains):
    if not selected_comparison_domains: return None
    filtered = df[df['domain_category'].isin(selected_comparison_domains)]
    top_v = filtered['violation_name'].value_counts().head(5).index
    comp_df = filtered[filtered['violation_name'].isin(top_v)]
    grouped = comp_df.groupby(['domain_category', 'violation_name']).size().reset_index(name='count')
    return px.bar(grouped, x='violation_name', y='count', color='domain_category', barmode='group')

def plot_violation_treemap(df):
    """
    Visualizes the hierarchy of violations: 
    Domain -> Category -> Specific Violation
    By setting color='violation_name', every 'leaf' and its parent categories 
    will be filled with diverse colors from the Plotly palette.
    """
    # Aggregate data for the treemap
    treemap_df = df.groupby(['domain_category', 'violation_category', 'violation_name']).size().reset_index(name='count')
    
    # Create the Treemap
    fig = px.treemap(
        treemap_df, 
        path=[px.Constant("All Domains"), 'domain_category', 'violation_category', 'violation_name'], 
        values='count',
        color='violation_name', 
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title="Hierarchical View of Accessibility Failures"
    )
    
    # Enhance the visual look (Clean "Card" style)
    fig.update_traces(
        textinfo="label+value",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
        marker=dict(line=dict(width=1, color='white')) # Adds white borders for clarity
    )
    
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        hoverlabel=dict(bgcolor="white", font_size=14),
        width=800, 
        height=650
    )
    
    return fig

@st.cache_resource
def train_severity_model(df):
    """
    Trains a model to predict violation_impact.
    Returns the model and the dictionary of fitted encoders.
    """
    # Select features and target
    features = ['domain_category', 'violation_category', 'violation_name']
    target = 'violation_impact'
    
    ml_df = df[features + [target]].dropna().copy()
    
    # Initialize and fit encoders for each categorical column
    le_dict = {}
    for col in ml_df.columns:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])
        le_dict[col] = le
        
    # Split features and target
    X = ml_df[features]
    y = ml_df[target]
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_dict

def get_risk_clusters(df):
    """
    Groups websites based on their violation profiles.
    """
    # Create a profile for each page: count how many errors of each category they have
    pivot_df = df.pivot_table(
        index='web_URL_id', 
        columns='violation_category', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Add a column for intensity
    pivot_df['total_violations'] = pivot_df.sum(axis=1)
    
    # Scale data for KMeans
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    
    # Cluster into 3 groups (Low, Medium, High Risk)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pivot_df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Rename clusters based on average violation volume for better UX
    cluster_order = pivot_df.groupby('Cluster')['total_violations'].mean().sort_values().index
    risk_mapping = {cluster_order[0]: "🟢 Low Risk", 
                    cluster_order[1]: "🟡 Moderate Risk", 
                    cluster_order[2]: "🔴 High Risk"}
    
    pivot_df['Risk Level'] = pivot_df['Cluster'].map(risk_mapping)
    
    return pivot_df.reset_index()

@st.cache_data
def convert_df_to_csv(df):
    # This converts the dataframe into a CSV string
    return df.to_csv(index=False).encode('utf-8')

# --- Main App ---
def main():
    local_css()
    df = load_and_preprocess_data()
    
    # --- Sidebar ---
    st.sidebar.header("📊 Filter Dashboard")
    all_domains = df['domain_category'].unique().tolist()
    selected_domains = st.sidebar.multiselect("Select Domains", all_domains, default=all_domains)
    
    filtered_df = df[df['domain_category'].isin(selected_domains)]

    # NEW: Download Center in Sidebar
    st.sidebar.divider()
    st.sidebar.subheader("💾 Download Center")
    st.sidebar.write("Export the current filtered analysis for your records.")
    
    csv_data = convert_df_to_csv(filtered_df)
    
    st.sidebar.download_button(
        label="📥 Download Data Report (CSV)",
        data=csv_data,
        file_name='accessguru_filtered_report.csv',
        mime='text/csv',
        help="Downloads the cleaned dataset based on your current domain filters."
    )

    with st.sidebar.expander("📖 About the Project"):
        st.write("""
            **AccessGuru** analyzes over 3,500 real-world web accessibility 
            violations across 448 websites. 
            
            Our goal is to uncover patterns of digital exclusion and provide 
            data-driven insights to help build a more inclusive web.
        """)

    with st.sidebar.expander("🏷️ Violation Categories"):
        st.markdown("""
            - **Syntactic:** Code-level errors (missing tags, improper ARIA).
            - **Semantic:** Meaning-level errors (missing alt-text, poor headings).
            - **Layout:** Visual-level errors (low color contrast, spacing).
        """)
    
    # Add a 'Clear Cache' button for the hackathon presentation
    if st.sidebar.button("Re-train ML Model"):
        st.cache_resource.clear()
        st.rerun()

    # --- Header ---
    col1, col2 = st.columns([1, 8])

    with col1:
        st.image("assets/icon.png", width=70)

    with col2:
        st.markdown(
            """
            <h2 style='color: #1f77b4; padding-top: 10px; margin-bottom: 0;'>
                Knoxicle: AccessGuru - Accessibility Insights
            </h2>
            """, 
            unsafe_allow_html=True
        )
        
    st.divider()
    st.markdown(
            """
            <h5 style='color: #1f77b4; padding-top: 10px; margin-bottom: 0;'>
                Analyzing Web Accessibility Violations (WCAG 2.1)
            </h5>
            """, 
            unsafe_allow_html=True
        )
    # st.markdown("Analyzing Web Accessibility Violations (WCAG 2.1)")

    # --- Kpi Row ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Violations", len(filtered_df))
    with kpi2:
        st.metric("Unique Pages", filtered_df['web_URL_id'].nunique())
    with kpi3:
        st.metric("Critical Errors", len(filtered_df[filtered_df['violation_impact'] == 'critical']))
    with kpi4:
        avg_score = filtered_df['violation_score'].mean()
        st.metric("Avg Severity", f"{avg_score:.2f}")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔥 Comparison", "🚨 Severe Issues", "🌲 Hierarchy", "🤖 ML Insights"
    ])

    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Violations by Domain")
            st.plotly_chart(plot_domain_rankings(filtered_df), use_container_width=True)
        with col_right:
            st.subheader("Violation Category Split")
            st.plotly_chart(plot_category_patterns(filtered_df), use_container_width=True)
        
        st.subheader("Most Frequent Violation Types")
        st.plotly_chart(plot_top_violation_types(filtered_df), use_container_width=True)

    with tab2:
        st.subheader("🕵️ Compare Invisible Barriers")
        comp_selection = st.multiselect("Pick domains to compare side-by-side:", all_domains, default=all_domains[:2])
        if comp_selection:
            st.plotly_chart(plot_invisible_barriers(df, comp_selection), use_container_width=True)
        
        st.divider()
        st.subheader("Violation Density Heatmap")
        st.plotly_chart(plot_heatmap(filtered_df), use_container_width=True)

    with tab3:
        st.subheader("🚨 Top 10 Inaccessible Pages")
        st.table(get_severe_pages(filtered_df))
        
        impact_colors = {
            "critical": "#8B0000",  # Dark Red
            "serious": "#E63946",   # Bright Red
            "moderate": "#F4A261",  # Orange
            "minor": "#2A9D8F"      # Teal/Green
        }
        # st.subheader("Impact Distribution")
        # st.plotly_chart(px.histogram(filtered_df, x='violation_impact', color='violation_impact', color_discrete_sequence=px.colors.qualitative.Safe), use_container_width=True)
        st.subheader("Impact Distribution")
        fig_impact = px.histogram(
            filtered_df, 
            x='violation_impact', 
            color='violation_impact', 
            color_discrete_map=impact_colors,
            # Keep the order logical: from most severe to least
            category_orders={"violation_impact": ["critical", "serious", "moderate", "minor"]},
            labels={'violation_impact': 'Impact Level', 'count': 'Number of Violations'}
        )
        
        fig_impact.update_layout(
            showlegend=True, 
            legend_title_text='Impact Level',
            xaxis_title="Violation Impact", 
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Moves legend to top
        )

        st.plotly_chart(fig_impact, use_container_width=True)

    with tab4:
        st.subheader("🌲 Global Violation Hierarchy")
        st.write("Click on a domain (e.g., 'tech') to see which violation categories and specific rules are failing most often.")
        
        # This will render with the high-contrast Plotly colors
        treemap_fig = plot_violation_treemap(filtered_df)
        st.plotly_chart(treemap_fig, use_container_width=True)
        
        st.info("💡 **Dashboard Tip:** The size of the boxes represents the volume of violations. This allows you to see at a glance that while one domain might have many pages, another might have a higher density of specific 'Semantic' or 'Layout' errors.")

    with tab5:
        st.header("🤖 Machine Learning & Risk Modeling")

        # Technical Logic Summary for Judges
        with st.expander("🛠️ How the AI Model Works (The Pipeline)"):
            st.write("""
                Our predictive engine follows a 4-step architecture:
                1. **Input:** User selects categorical web attributes.
                2. **Encoding:** Text labels are mapped to high-dimensional integers.
                3. **Classification:** A Random Forest model (100 estimators) evaluates the risk profile.
                4. **Decoding:** Numerical results are transformed back into WCAG Impact levels.
            """)

        # --- Prediction Section ---
        st.subheader("🔮 Predict Violation Impact")
        st.markdown("Select attributes to estimate how severe a web accessibility violation will be.")
        
        # Train the model inside main() and retrieve encoders
        model, le_dict = train_severity_model(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_domain = st.selectbox("Industry/Domain", le_dict['domain_category'].classes_)
        with col2:
            selected_cat = st.selectbox("Violation Category", le_dict['violation_category'].classes_)
        with col3:
            available_names = df[df['violation_category'] == selected_cat]['violation_name'].unique()
            selected_name = st.selectbox("Specific Violation", sorted(available_names))

        if st.button("Run AI Prediction"):
            # Convert strings back to numbers using the encoders
            input_data = pd.DataFrame([{
                'domain_category': le_dict['domain_category'].transform([selected_domain])[0],
                'violation_category': le_dict['violation_category'].transform([selected_cat])[0],
                'violation_name': le_dict['violation_name'].transform([selected_name])[0]
            }])
            
            # Make Prediction
            pred_num = model.predict(input_data)[0]
            impact_result = le_dict['violation_impact'].inverse_transform([pred_num])[0]
            
            # Define icons for the impact levels for extra visual flair to display the attributes
            impact_icons = {
                "critical": "🚨",
                "serious": "🛑",
                "moderate": "⚠️",
                "minor": "ℹ️"
            }
            icon = impact_icons.get(impact_result.lower(), "✨")

            st.info(f"""
                ### {icon} Prediction Result: **{impact_result.upper()}**
                
                **Analysis Summary:**
                - **Domain:** {selected_domain}
                - **Violation Category:** {selected_cat}
                - **Violation Name:** `{selected_name}`
                
                The AI model determined that this specific configuration typically results in a **{impact_result}** impact on digital accessibility.
            """)

        st.divider()

        # --- Clustering Section ---
        st.subheader("🛡️ Website Risk Clustering")
        st.write("We used K-Means Clustering to group the 448 websites by their 'failure patterns'.")
        
        # Generate the cluster data
        clusters = get_risk_clusters(df)
        
        # We detect which category columns exist in the processed cluster dataframe
        possible_categories = ['Layout', 'Semantic', 'Syntax']
        available_categories = [col for col in possible_categories if col in clusters.columns]

        selected_y = st.selectbox(
            "Select Violation Category for Y-Axis Analysis:", 
            options=available_categories,
            index=0  # Default to 'Layout'
        )
        
        # Create the interactive scatter plot using the selection
        fig_clusters = px.scatter(
            clusters, 
            x="total_violations", 
            y=selected_y, 
            color="Risk Level",
            hover_data=["web_URL_id"],
            title=f"Clustering Sites: Total Errors vs. {selected_y} Failures",
            labels={
                "total_violations": "Total Violations per Site",
                selected_y: f"{selected_y} Violations"
            },
            color_discrete_map={
                "🟢 Low Risk": "green", 
                "🟡 Moderate Risk": "orange", 
                "🔴 High Risk": "red"
            }
        )
        
        # Enhance plot aesthetics
        fig_clusters.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        
        st.plotly_chart(fig_clusters, use_container_width=True)

        st.info(f"""
            **Insight:** By switching to **{selected_y}**, you can see if the 'High Risk' cluster 
            is driven specifically by that type of violation or if it's a general spread across all categories.
        """)

        # --- Domain Ranking ---
        st.subheader("🏆 Ranking of Inaccessible Design")
        st.write("Rank domains based on the likelihood of encountering specific impact levels.")

        # Add the Selector
        impact_to_rank = st.selectbox(
            "Select Impact Level to Rank By:",
            options=["critical", "serious", "moderate", "minor"],
            index=0  # Default to Critical
        )

        # Calculate the Likelihood (Percentage) dynamically
        # We group by domain and calculate how often the selected impact occurs
        ranking = df.groupby('domain_category').apply(
            lambda x: (x['violation_impact'] == impact_to_rank).mean() * 100
        ).reset_index(name='Likelihood %').sort_values('Likelihood %', ascending=False)

        # Display with the Progress Column
        st.dataframe(
            ranking,
            column_config={
                "domain_category": "Domain",
                "Likelihood %": st.column_config.ProgressColumn(
                    f"Likelihood of {impact_to_rank.title()} Issues",
                    help=f"Percentage of violations in this domain that are {impact_to_rank}",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True,
            use_container_width=True
        )

        # st.caption(f"**Interpretation:** In the top-ranked domain, nearly {ranking['Likelihood %'].iloc[0]:.1f}% of all detected violations are classified as **{impact_to_rank}**.")
        top_domain = ranking['domain_category'].iloc[0]
        top_value = ranking['Likelihood %'].iloc[0]

        st.info(f"""
            ### 📊 Risk Summary: {impact_to_rank.title()} Impact
            
            **Key Findings:**
            - **Highest Risk Domain:** {top_domain.title()}
            - **Concentration:** {top_value:.1f}% of violations in this domain are {impact_to_rank}.
            
            **Digital Equity Insight:**
            When a domain has a high concentration of **{impact_to_rank}** issues, it indicates a systemic failure in the 
            design process of that industry. For users, this means the barrier isn't just a one-off mistake, but a 
            pattern that makes these types of sites (like {top_domain.title()}) fundamentally harder to access.
        """)

    # --- Footer ---
    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    
    with footer_col1:
        st.markdown("""
            **Data Source:** AccessGuru Dataset (WCAG 2.1 Guidelines)  
            *3,500+ violations across Health, Education, Gov, and more.*
        """)
        
    with footer_col2:
        st.markdown("""
            <div style='text-align: right;'>
                Created for the <b>DubsTech Data Datathon 2026</b><br>
                Team: Knoxicle
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()