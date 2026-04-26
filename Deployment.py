import streamlit as st
import pandas as pd
import joblib
import pickle
import altair as alt
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load the Saved Model, Encoders, and Data ---
@st.cache_resource  # Use cache_resource for model/encoders
def load_assets():
    model = joblib.load('college_predictor_compressed.joblib')
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Load raw data for the dashboard and dropdowns
    df_2023 = pd.read_csv('josaa_2023_all_institutes_full.csv')
    df_2024 = pd.read_csv('josaa_2024_reparsed_all_institutes.csv')
    df_2025 = pd.read_csv('josaa_2025_opening_closing_ranks_all_institutes.csv')
    
    df_2023['Year'] = 2023
    df_2024['Year'] = 2024
    df_2025['Year'] = 2025
    
    df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
    df.dropna(subset=['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender'], inplace=True)
    
    df['Opening Rank'] = pd.to_numeric(df['Opening Rank'].astype(str).str.replace('P', ''), errors='coerce')
    df['Closing Rank'] = pd.to_numeric(df['Closing Rank'].astype(str).str.replace('P', ''), errors='coerce')
    df.dropna(subset=['Opening Rank', 'Closing Rank'], inplace=True)
    
    return model, encoders, df

model, encoders, df = load_assets()


# --- 2. Create the User Interface ---
st.set_page_config(page_title="JoSAA 2026 College Predictor", layout="wide", initial_sidebar_state="expanded")

# Premium CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🎓 JoSAA 2026: Advanced College Predictor')
st.markdown("---")

# 1. Select Institute Category
inst_type = st.radio("Select Institute Category", ["IIT (Indian Institute of Technology)", "Non-IIT (NITs, IIITs, GFTIs)"], horizontal=True)

# Filter dataframe based on selection
if "Non-IIT" in inst_type:
    # Everything EXCEPT pure IITs
    # Pure IITs contain "Indian Institute of Technology" but NOT "Information"
    mask = df['Institute'].str.contains('Indian Institute of Technology') & ~df['Institute'].str.contains('Information')
    display_df = df[~mask]
else:
    # Pure IITs only
    mask = df['Institute'].str.contains('Indian Institute of Technology') & ~df['Institute'].str.contains('Information')
    display_df = df[mask]

col1, col2 = st.columns(2)
with col1:
    st.header("Personal Details")
    user_rank = st.number_input('Enter your Category Rank', min_value=1, max_value=400000, value=10000, step=100)
    
    # Use the filtered display_df for the dropdowns
    available_institutes = sorted(display_df['Institute'].unique())
    selected_institute = st.selectbox('Select Institute', available_institutes)
    
    df_filtered_1 = display_df[display_df['Institute'] == selected_institute]
    available_programs = sorted(df_filtered_1['Academic Program Name'].unique())
    selected_program = st.selectbox('Select Academic Program', available_programs)

with col2:
    st.header("Preferences")
    df_filtered_2 = df_filtered_1[df_filtered_1['Academic Program Name'] == selected_program]
    selected_quota = st.selectbox('Select Quota', sorted(df_filtered_2['Quota'].unique()))
    df_filtered_3 = df_filtered_2[df_filtered_2['Quota'] == selected_quota]
    selected_seat_type = st.selectbox('Select Seat Type', sorted(df_filtered_3['Seat Type'].unique()))
    df_filtered_4 = df_filtered_3[df_filtered_3['Seat Type'] == selected_seat_type]
    selected_gender = st.selectbox('Select Gender', sorted(df_filtered_4['Gender'].unique()))

if st.button('Predict My Chance', type="primary"):
    try:
        input_data = {
            'Institute': encoders['Institute'].transform([selected_institute])[0],
            'Academic Program Name': encoders['Academic Program Name'].transform([selected_program])[0],
            'Quota': encoders['Quota'].transform([selected_quota])[0],
            'Seat Type': encoders['Seat Type'].transform([selected_seat_type])[0],
            'Gender': encoders['Gender'].transform([selected_gender])[0],
            'Year': 2026
        }
        input_df = pd.DataFrame([input_data])
        predicted_rank = model.predict(input_df)[0]
        predicted_rank = int(round(predicted_rank))

        st.markdown("### Prediction Result")
        if user_rank <= predicted_rank:
            st.success(f"🎉 **High Chance!** Predicted Closing Rank: **{predicted_rank:,}**")
        else:
            st.error(f"😞 **Low Chance.** Predicted Closing Rank: **{predicted_rank:,}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# --- 4. Fully Interactive Data Dashboard ---
st.header("Interactive Data Dashboard (Based on 2023-2025 Data)")
with st.expander("Click to see visualizations"):

    # --- First Row of Dashboard ---
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # --- Plot 1: Top 20 Institutes (ALTAIR) ---
        st.subheader("Top 20 Institutes")
        institute_counts = df['Institute'].value_counts().nlargest(20)
        plot_data_inst = institute_counts.reset_index()
        plot_data_inst.columns = ['Institute', 'Number of Entries']
        
        chart_inst = alt.Chart(plot_data_inst).mark_bar().encode(
            x=alt.X('Number of Entries:Q'),
            y=alt.Y('Institute:N', sort='-x'), 
            tooltip=['Institute', 'Number of Entries']
        ).properties(
            title='Top 20 Institutes by Program Entries'
        ).interactive() 
        
        st.altair_chart(chart_inst, use_container_width=True)

    with viz_col2:
        # --- Plot 2: Seat Type Distribution (ALTAIR) ---
        st.subheader("Seat Type Distribution")
        seat_type_counts = df['Seat Type'].value_counts()
        
        if len(seat_type_counts) > 10:
            top_10 = seat_type_counts.nlargest(9)
            other_sum = seat_type_counts.nsmallest(len(seat_type_counts) - 9).sum()
            other = pd.Series([other_sum], index=['Other'])
            plot_data_seat = pd.concat([top_10, other])
        else:
            plot_data_seat = seat_type_counts
        
        plot_data_df = plot_data_seat.reset_index()
        plot_data_df.columns = ['Seat Type', 'Count']
        plot_data_df['Percentage'] = (plot_data_df['Count'] / plot_data_df['Count'].sum())
        plot_data_df['LegendLabel'] = plot_data_df['Seat Type'] + ' - ' + \
                                      (plot_data_df['Percentage'] * 100).round(1).astype(str) + '%'

        base = alt.Chart(plot_data_df).encode(
           theta=alt.Theta("Count", stack=True)
        ).properties(
            title='Overall Distribution of Seat Types'
        )
        
        pie = base.mark_arc(outerRadius=140).encode(
            color=alt.Color("LegendLabel", title="Seat Type"), 
            order=alt.Order("Count", sort="descending"),
            tooltip=["Seat Type", "Count", alt.Tooltip("Percentage", format=".1%")]
        )
        
        chart_seat = pie 
        st.altair_chart(chart_seat)

    st.divider() # Add a visual separator

    # --- Second Row of Dashboard ---
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        # --- PLOT 3 (NEW): Gender Distribution (ALTAIR) ---
        st.subheader("Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        plot_data_gender = gender_counts.reset_index()
        plot_data_gender.columns = ['Gender', 'Count']
        plot_data_gender['Percentage'] = (plot_data_gender['Count'] / plot_data_gender['Count'].sum())
        plot_data_gender['LegendLabel'] = plot_data_gender['Gender'] + ' - ' + \
                                          (plot_data_gender['Percentage'] * 100).round(1).astype(str) + '%'

        base_gender = alt.Chart(plot_data_gender).encode(
           theta=alt.Theta("Count", stack=True)
        ).properties(title='Overall Gender Distribution')

        pie_gender = base_gender.mark_arc(outerRadius=140).encode(
            color=alt.Color("LegendLabel", title="Gender"),
            order=alt.Order("Count", sort="descending"),
            tooltip=["Gender", "Count", alt.Tooltip("Percentage", format=".1%")]
        )
        
        chart_gender = pie_gender
        st.altair_chart(chart_gender)

    with viz_col4:
        # --- PLOT 4 (NEW): Quota Distribution (ALTAIR) ---
        st.subheader("Quota Distribution")
        quota_counts = df['Quota'].value_counts()
        plot_data_quota = quota_counts.reset_index()
        plot_data_quota.columns = ['Quota', 'Number of Entries']

        chart_quota = alt.Chart(plot_data_quota).mark_bar().encode(
            x=alt.X('Quota:N', sort='-y'), # Sort by count
            y=alt.Y('Number of Entries:Q'),
            color=alt.Color("Quota", title="Quota"),
            tooltip=['Quota', 'Number of Entries']
        ).properties(
            title='Entries by Quota'
        ).interactive()

        st.altair_chart(chart_quota, use_container_width=True)

    
