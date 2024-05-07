import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Start ------------------------------------------------------------------------------------------- Loading File
# Define the file path
dt = pd.read_csv("archive/mental-heath-in-tech-2016_20161114.csv")
dt_rows, dt_columns = dt.shape

# get survey answers
print(f'Dataset Loaded: \n{dt.head()}')

def show_survey_questions():
    for i, column in enumerate(dt.columns):
        print(str(i) + " " + str(column))

mission_data_list = dt.isna().sum().tolist()
print(f"Mission data in columns: {mission_data_list}")

def show_missing_values(dt, more_than_p=0.3):
    dt_rows, _ = dt.shape
    mission_data_list = dt.isna().sum().tolist()
    more_than_n = dt_rows*more_than_p
    for i, n in enumerate(mission_data_list):
        if mission_data_list[i] >= more_than_n:
            print(f'{n / dt_rows: .2f}% Missing -> ({i}/{dt[dt.columns[i]].dtype}) {dt.columns[i]}')

show_missing_values(dt, more_than_p=0.26)
# Finish ------------------------------------------------------------------------------------------- Loading File

# Start ------------------------------------------------------------------------------------------- Dataset Preprocessing
# Cleaning Age Feature
filtered_values = dt[(dt['What is your age?'] > 99) | (dt['What is your age?'] < 18)]
print(f'Found in "What is your age?" {len(filtered_values)} incorrect data ')
filtered_indices = filtered_values.index
valid_age_indices = dt.index.difference(filtered_indices)
mean_age = dt.loc[valid_age_indices, 'What is your age?'].mean()
dt.loc[filtered_indices, 'What is your age?'] = int(mean_age)
print(f"Updated the {filtered_indices} with the int mean of others({int(mean_age)})")

# Cleaning Gender Feature
unique_values_gender_before = dt['What is your gender?'].unique()
print(f"The gender feature have {len(unique_values_gender_before)} unique values")
def clean_gender(gender):
    gender = str(gender).strip().lower()
    if gender in ['male', 'm', 'man', 'cis male', 'malr', 'mail', 'male.', 'sex is male', 'm|', 'cis male', 'cis man', 'male (cis)', 'dude']:
        return 'Male'
    elif gender in ['female', 'f', 'woman', 'cis female', 'cis-woman', 'cisgender female', 'female assigned at birth', 'fem', 'female.', 'female ']:
        return 'Female'
    else:
        return 'Other'

dt['What is your gender?'] = dt['What is your gender?'].apply(clean_gender)
print(f"After processing gender feature have 3 unique values: {dt['What is your gender?'].unique()}")


# Cleaning not relevant data
def show_data_to_be_removed_relevant_info(feature_key, expected_value):
    dataset_len = len(dt[feature_key])
    dataset_interested_in_len = len(dt.loc[(dt[feature_key] == expected_value)])
    print(f'Feature "{feature_key}" have unique values: {dt[feature_key].unique()}')
    print(f'Interested dat values is "{expected_value}"')
    print(f'The drop in percentage is {1 - dataset_interested_in_len/dataset_len: .2f}%')

for feature_key, expected_value in [('Are you self-employed?', 0), ('Is your employer primarily a tech company/organization?', 1)]:
    show_data_to_be_removed_relevant_info(feature_key, expected_value)

# Select only participants working for a company
dt = dt.loc[(dt['Are you self-employed?'] == 0)]
dt = dt.reset_index(drop=True)

# Select only participants working for a tech company
dt = dt.loc[(dt['Is your employer primarily a tech company/organization?'] == 1)]
dt = dt.reset_index(drop=True)




#Create separate columns for presence of each MHC for easier filter
details_about_condition_feature_key = 'If yes, what condition(s) have you been diagnosed with?'
print(dt[details_about_condition_feature_key].value_counts())
dt['Anxiety Disorder'] = dt[details_about_condition_feature_key].str.contains('Anxiety Disorder')
dt['Mood Disorder'] = dt[details_about_condition_feature_key].str.contains('Mood Disorder')
dt['ADHD'] = dt[details_about_condition_feature_key].str.contains('Attention')
dt['OCD'] = dt[details_about_condition_feature_key].str.contains('Compulsive')
dt['PTSD'] = dt[details_about_condition_feature_key].str.contains('Post')
dt['PTSD undiagnosed'] = dt[details_about_condition_feature_key].str.contains('PTSD \(undiagnosed\)')
dt['Eating Disorder'] = dt[details_about_condition_feature_key].str.contains('Eating')
dt['Substance Use Disorder'] = dt[details_about_condition_feature_key].str.contains('Substance')
dt['Stress Response Syndrome'] = dt[details_about_condition_feature_key].str.contains('Stress Response')
dt['Personality Disorder'] = dt[details_about_condition_feature_key].str.contains('Personality Disorder')
dt['Pervasive Developmental Disorder'] = dt[details_about_condition_feature_key].str.contains('Pervasive')
dt['Psychotic Disorder'] = dt[details_about_condition_feature_key].str.contains('Psychotic')
dt['Addictive Disorder'] = dt[details_about_condition_feature_key].str.contains('Addictive Disorder')
dt['Dissociative Disorder'] = dt[details_about_condition_feature_key].str.contains('Dissociative')
dt['Seasonal Affective Disorder'] = dt[details_about_condition_feature_key].str.contains('Seasonal')
dt['Schizotypal Personality Disorder'] = dt[details_about_condition_feature_key].str.contains('Schizotypal')
dt['Traumatic Brain Injury'] = dt[details_about_condition_feature_key].str.contains('Brain')
dt['Sexual Addiction'] = dt[details_about_condition_feature_key].str.contains('Sexual')
dt['Autism'] = dt[details_about_condition_feature_key].str.contains('Autism')
dt['ADD w/o Hyperactivity)'] = dt[details_about_condition_feature_key].str.contains('ADD \(w/o Hyperactivity\)')

# Finish ------------------------------------------------------------------------------------------- Dataset Preprocessing


# Start ------------------------------------------------------------------------------------------- Dataset Visualizing

currently_have_mental_health_disorder_feature_counts = dt['Do you currently have a mental health disorder?'].value_counts()
print(currently_have_mental_health_disorder_feature_counts)
# Plot
plt.figure(figsize=(6, 6))
plt.pie(
    currently_have_mental_health_disorder_feature_counts.values.tolist(),
    labels=currently_have_mental_health_disorder_feature_counts.index.tolist(),
    autopct='%1.1f%%',
    startangle=140
)
plt.title('Do you currently have a mental health disorder?')
plt.xlabel('Response')
plt.ylabel('Count')
plt.savefig('mental_health_disorder_plot.png')


# Splitting to smaller datasets
# dt_no_MHD = dt.loc[pd.isna(dt[details_about_condition_feature_key])]
dt_no_MHD = dt.loc[(dt['Do you currently have a mental health disorder?'] == "No")]
dt_yes_MHD = dt.loc[(dt['Do you currently have a mental health disorder?'] == "Yes")]
dt_maybe_MHD = dt.loc[(dt['Do you currently have a mental health disorder?'] == "Maybe")]
dt_anx_dep = dt.loc[(dt['Anxiety Disorder'] == 1) & (dt['Mood Disorder'] == 1)]
dt_adhd = dt.loc[(dt['ADHD'] == 1)]
dt_ocd = dt.loc[(dt['OCD'] == 1)]
dt_ptsd = dt.loc[(dt['PTSD'] == 1)]

#Display population size of each group
total_population_len = len(dt)
dt_group = [
    (dt, 'Whole Population'),
    (dt_no_MHD, 'No MHC'),
    (dt_yes_MHD, 'MHC'),
    (dt_maybe_MHD, 'MHC Maybe'),
    (dt_anx_dep, 'Anxiety & Depression'),
    (dt_adhd, 'ADHD'),
    (dt_ptsd, 'PTSD'),
    (dt_ocd, 'OCD')
]
dt_group_labels = [label for _, label in dt_group]
dt_group_datasets = [data for data, _ in dt_group]
dt_group_datasets_counts = [len(data) for data in dt_group_datasets]
percentages = [(len(data) / total_population_len) * 100 for data in dt_group_datasets]
dt_MHD_count = pd.DataFrame({
    'Population': dt_group_labels,
    'Count': dt_group_datasets_counts,
    'Percentage': percentages
})

# Plot dt_MHD_count DataFrame
plt.figure(figsize=(14, 6))
bars = plt.bar(dt_group_labels, dt_group_datasets_counts, color='skyblue')
for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f'{percentage:.2f}%', ha='center', color='black')
plt.title('Population Distribution of Mental Health Conditions')
plt.xlabel('Mental Health Conditions')
plt.ylabel('Number of Individuals')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('population_count_plot.png')


def multiply_tuple_scalar(tuple_to_multiply, scalar):
    return tuple(value * scalar for value in tuple_to_multiply)

# Gender statistics
def create_sorted_dt(dt, label, base_label):
    counts = dt[base_label].value_counts().sort_index().reset_index()
    counts.columns = [base_label, label]
    return counts

dt_gender = None
for data, label in dt_group:
    counts_MHC = create_sorted_dt(data, f'What is your gender? ({label})', base_label='What is your gender?')
    dt_gender = dt_gender.merge(counts_MHC, on='What is your gender?', how='left') if dt_gender is not None else counts_MHC

# Plotting
def plot_dt_group_statistics(dt_statistics, statistics_feature, title, filename, save_format='png', figsize=1):
    fig, axs = plt.subplots(2, 4, figsize=multiply_tuple_scalar((12, 8), figsize))
    axs = axs.flatten()  # Flatten the subplot array for easier iteration

    colors = ['skyblue', 'salmon', 'lightgreen', 'orchid', 'gold']
    # MHC_labels = [whole_population] + MHCs
    MHC_columns = dt_statistics.columns.to_list()[1:]
    # Plot each MHC distribution
    for ax, label, column in zip(axs, dt_group_labels, MHC_columns):
        ax.pie(dt_statistics[column], labels=dt_statistics[statistics_feature], colors=colors, autopct='%.0f%%')
        ax.set_title(label, fontsize=12)

    # Adjust layout and add title
    plt.suptitle(title, weight='bold', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout to make space for title

    plt.savefig(f'{filename}.{save_format}')

plot_dt_group_statistics(
    dt_statistics=dt_gender,
    statistics_feature='What is your gender?',
    title='Gender Distribution by Mental Health Condition',
    filename="population_count_by_gender_plot"
)


# Age statistics
age_stats = {}
for data, label in dt_group:
    age_stats[label] = data['What is your age?'].describe()
dt_age_stats = pd.DataFrame(age_stats).reset_index()
dt_age_stats.columns = ['Statistic'] + dt_group_labels
print(dt_age_stats)


# sought treatment statistics
sought_treatment_feature = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
dt_treated_sought_counts = None
for data, label in dt_group:
    counts_MHC = create_sorted_dt(data, label, base_label=sought_treatment_feature)
    dt_treated_sought_counts = dt_treated_sought_counts.merge(counts_MHC, on=sought_treatment_feature, how='left') if dt_treated_sought_counts is not None else counts_MHC

# Plotting sought treatment statistics
plot_dt_group_statistics(
    dt_statistics=dt_treated_sought_counts,
    statistics_feature=sought_treatment_feature,
    title='Treatment Sought Distribution by Mental Health Condition',
    filename="population_count_by_treatment_sought_plot"
)




# work interfere statistics
mhc_interferes_with_work_feature = 'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'
dt_mhc_interferes_with_work_feature_counts = None
for data, label in dt_group:
    counts_MHC = create_sorted_dt(data, label, base_label=mhc_interferes_with_work_feature)
    dt_mhc_interferes_with_work_feature_counts = dt_mhc_interferes_with_work_feature_counts.merge(counts_MHC, on=mhc_interferes_with_work_feature, how='left') if dt_mhc_interferes_with_work_feature_counts is not None else counts_MHC
dt_mhc_interferes_with_work_feature_counts = dt_mhc_interferes_with_work_feature_counts.fillna(0)
# Plotting work interfere statistics
plot_dt_group_statistics(
    dt_statistics=dt_mhc_interferes_with_work_feature_counts,
    statistics_feature=mhc_interferes_with_work_feature,
    title='Work Interferes Distribution by Mental Health Condition',
    filename="population_count_by_mhc_work_interferes_plot",
    figsize=1.5
)



# work interfere not treated effectively statistics
work_interfere_not_treated_effectively_feature = 'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'
work_interfere_not_treated_effectively_feature_counts = None
for data, label in dt_group:
    counts_MHC = create_sorted_dt(data, label, base_label=work_interfere_not_treated_effectively_feature)
    work_interfere_not_treated_effectively_feature_counts = work_interfere_not_treated_effectively_feature_counts.merge(counts_MHC, on=work_interfere_not_treated_effectively_feature, how='left') if work_interfere_not_treated_effectively_feature_counts is not None else counts_MHC
work_interfere_not_treated_effectively_feature_counts = work_interfere_not_treated_effectively_feature_counts.fillna(0)
# Plotting work interfere statistics
plot_dt_group_statistics(
    dt_statistics=work_interfere_not_treated_effectively_feature_counts,
    statistics_feature=work_interfere_not_treated_effectively_feature,
    title='Work Interferes Not Treated Effectively Distribution by Mental Health Condition',
    filename="population_count_by_work_interfere_not_treated_effectively_plot",
    figsize=1.5
)


# Finish ------------------------------------------------------------------------------------------- Dataset Visualizing

# Start ------------------------------------------------------------------------------------------- Filling None values
# Finish ------------------------------------------------------------------------------------------- Filling None values



