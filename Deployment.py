import streamlit as st
import pandas as pd
import pickle
import altair as alt  # The main visualization library
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load the Saved Model, Columns, and Data ---
@st.cache_data  # Add caching to speed up data loading
def load_data():
    with open('college_predictor.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_columns.pkl', 'rb') as file:
        model_columns = pickle.load(file)

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
    return model, model_columns, df

model, model_columns, df = load_data()


# --- 2. Create the User Interface ---
st.set_page_config(page_title="JEE College Predictor", layout="wide")
st.title('🎓 JoSAA 2026: College Predictor')
st.markdown("Select your preferences to predict the closing rank based on historical JOSAA data.")

institutes = sorted(df['Institute'].unique())
col1, col2 = st.columns(2)
with col1:
    st.header("Your Details")
    user_rank = st.number_input('Enter your JEE Rank as per your category', min_value=1, max_value=400000, value=10000, step=100)
    selected_institute = st.selectbox('Select Institute', institutes)
    df_filtered_1 = df[df['Institute'] == selected_institute]
    available_programs = sorted(df_filtered_1['Academic Program Name'].unique())
    selected_program = st.selectbox('Select Academic Program', available_programs)

with col2:
    st.header("Category and Quota")
    df_filtered_2 = df_filtered_1[df_filtered_1['Academic Program Name'] == selected_program]
    available_quotas = sorted(df_filtered_2['Quota'].unique())
    selected_quota = st.selectbox('Select Quota', available_quotas)
    df_filtered_3 = df_filtered_2[df_filtered_2['Quota'] == selected_quota]
    available_seat_types = sorted(df_filtered_3['Seat Type'].unique())
    selected_seat_type = st.selectbox('Select Seat Type', available_seat_types)
    df_filtered_4 = df_filtered_3[df_filtered_3['Seat Type'] == selected_seat_type]
    available_genders = sorted(df_filtered_4['Gender'].unique())
    selected_gender = st.selectbox('Select Gender', available_genders)

# --- 3. Make Prediction ---
if st.button('Predict Closing Rank', type="primary"):
    input_df = pd.DataFrame(columns=model_columns, index=[0])
    input_df.fillna(0, inplace=True)
    input_Dyear = df['Year'].max() + 1
    if 'Year' in input_df.columns:
        input_df['Year'] = input_Dyear
    if f'Institute_{selected_institute}' in input_df.columns:
        input_df[f'Institute_{selected_institute}'] = 1
    if f'Academic Program Name_{selected_program}' in input_df.columns:
        input_df[f'Academic Program Name_{selected_program}'] = 1
    if f'Quota_{selected_quota}' in input_df.columns:
        input_df[f'Quota_{selected_quota}'] = 1
    if f'Seat Type_{selected_seat_type}' in input_df.columns:
        input_df[f'Seat Type_{selected_seat_type}'] = 1
    if f'Gender_{selected_gender}' in input_df.columns:
        input_df[f'Gender_{selected_gender}'] = 1

    predicted_rank = model.predict(input_df)[0]
    predicted_rank = int(round(predicted_rank))

    st.subheader('Prediction Result')
    if user_rank <= predicted_rank:
        st.success(f"🎉 **High Chance!**")
        st.markdown(f"The predicted closing rank is **{predicted_rank:,}**. Your rank of **{user_rank:,}** is within the predicted range.")
    else:
        st.error(f"😞 **Low Chance.**")
        st.markdown(f"The predicted closing rank is **{predicted_rank:,}**. Your rank of **{user_rank:,}** is higher than the predicted closing rank.")

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

    
