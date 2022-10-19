import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import urllib.request
import requests
import json

st.set_page_config(layout='wide')
st.title('Table Format Projects on Github')
st.markdown("""
This app gets activity data from Github for three projects: [Delta Lake](https://github.com/delta-io), [Apache Hudi](https://github.com/apache/hudi), and [Apache Iceberg](https://github.com/apache/iceberg)
""")

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Contributions','Pull Requests','Commits'])

# GET DATA
## PROJECTS & CONTRIBUTORS
@st.experimental_memo(ttl=60*60*24*30)
def load_projects():
    with urllib.request.urlopen('https://api.github.com/orgs/delta-io/repos') as response:
        delta_json_obj = json.load(response)
    delta_proj_repos = []
    for i in delta_json_obj:
        delta_proj_repos.append(i['full_name'])
        
    all_proj_repos = ['apache/iceberg','apache/hudi']
    all_proj_repos.extend(delta_proj_repos)

    return all_proj_repos

all_proj_repos = load_projects()

@st.experimental_memo(ttl=60*60*24*30)
def load_project_contributors():
    all_proj_contributors = []
    for i in all_proj_repos:
        all_proj_contributors.append({'repo':str(i), 'contributors':''})
    
    for i in all_proj_contributors:
        url = 'https://api.github.com/repos/'+str(i['repo'])+'/contributors?per_page=100&page=1'
        response = requests.get(url,headers={"Authorization": 'token '+st.secrets["token_1"]})
        i['contributors'] = response.json()
        while 'next' in response.links.keys():
            response = requests.get(response.links['next']['url'],headers={"Authorization": 'token '+st.secrets["token_1"]})
            i['contributors'].extend(response.json())
    
    df_all_proj_contributors = pd.json_normalize([user for user in all_proj_contributors],
                 record_path = ['contributors'],
                 meta = ['repo'],
                 errors = 'ignore'
                 )[['login','url','contributions','repo']]
    
    conditions = [
        (df_all_proj_contributors['repo'].str.contains('delta-io') ),
        (df_all_proj_contributors['repo'].str.contains('iceberg') ),
        (df_all_proj_contributors['repo'].str.contains('hudi') )
    ]

    values = ['delta','iceberg','hudi']
    
    df_all_proj_contributors['project'] = np.select(conditions, values)

    return df_all_proj_contributors

df_all_proj_contributors = load_project_contributors()

# CONTRIBUTOR PROFILE DATA
@st.experimental_memo(ttl=60*60*24*30)
def load_unique_profiles():
    df_unique_contributors = df_all_proj_contributors['url'].unique()
    unique_profiles = []
    for i in df_unique_contributors:
        req = urllib.request.Request(i)
        req.add_header('Authorization', 'token '+st.secrets["token_1"])
        #progress_contributors.progress(j+1)
        with urllib.request.urlopen(req) as response:
                unique_profiles.append(json.load(response))
    
    df_unique_profiles = pd.DataFrame(unique_profiles, columns = ['login','company','email','created_at','updated_at'])
    
    return df_unique_profiles

df_unique_profiles = load_unique_profiles()

# LOAD PULL REQUEST DATA
@st.experimental_memo(ttl=60*60*24*30)
def load_project_pulls():
    all_proj_pulls = []
    for i in all_proj_repos:
        all_proj_pulls.append({'repo':str(i), 'pulls':''})
    
    for i in all_proj_pulls:
        url = 'https://api.github.com/repos/'+str(i['repo'])+'/pulls?state=all&page=1'
        response = requests.get(url,headers={"Authorization": 'token '+st.secrets["token_2"]})
        i['pulls'] = response.json()
        while 'next' in response.links.keys():
            response = requests.get(response.links['next']['url'],headers={"Authorization": 'token '+st.secrets["token_2"]})
            i['pulls'].extend(response.json())
    
    # flatten the JSON payload into a dataframe, only keeping fields needed
    df_pulls = pd.json_normalize([pull for pull in all_proj_pulls],
                            record_path = ['pulls'],
                            meta = ['repo'])[['repo','id','number','state','user.login','created_at','updated_at','closed_at','merged_at']]
    
    df_pulls = df_pulls.rename(columns={'user.login':'login'})
    
    conditions = [
        (df_pulls['repo'].str.contains('delta-io') ),
        (df_pulls['repo'].str.contains('iceberg') ),
        (df_pulls['repo'].str.contains('hudi') )
    ]

    values = ['delta','iceberg','hudi']
    
    df_pulls['project'] = np.select(conditions, values)
    
    return df_pulls

df_pulls = load_project_pulls()

# LOAD COMMITS DATA
@st.experimental_memo(ttl=60*60*24*30)
def load_project_commits():
    all_proj_commits = []
    for i in all_proj_repos:
        all_proj_commits.append({'repo':str(i), 'commits':''})
    
    for i in all_proj_commits:
        url = 'https://api.github.com/repos/'+str(i['repo'])+'/commits?state=all&page=1'
        response = requests.get(url,headers={"Authorization": 'token '+st.secrets["token_2"]})
        i['commits'] = response.json()
        while 'next' in response.links.keys():
            response = requests.get(response.links['next']['url'],headers={"Authorization": 'token '+st.secrets["token_2"]})
            i['commits'].extend(response.json())
    
    # flatten the JSON payload into a dataframe, only keeping fields needed
    df_commits = pd.json_normalize([commit for commit in all_proj_commits],
                            record_path = ['commits'],
                            meta = ['repo'])[['repo','commit.author.date','author.login']]
    
    df_commits = df_commits.rename(columns={'author.login':'login'})
    df_commits = df_commits.rename(columns={'commit.author.date':'created_at'})
    
    conditions = [
        (df_commits['repo'].str.contains('delta-io') ),
        (df_commits['repo'].str.contains('iceberg') ),
        (df_commits['repo'].str.contains('hudi') )
    ]

    values = ['delta','iceberg','hudi']
    
    df_commits['project'] = np.select(conditions, values)

    return df_commits
df_commits = load_project_commits()

# Clean company column
## Lowercase
df_unique_profiles['company'] = df_unique_profiles['company'].str.lower()

## Define conditions
company_patterns = [
    (df_unique_profiles['company'].str.contains('databricks', na=False, regex=False), 'databricks'),
    (df_unique_profiles['company'].str.contains('tabular', na=False, regex=False), 'tabular'),
    (df_unique_profiles['company'].str.contains('apple', na=False, regex=False), 'apple'),
    (df_unique_profiles['company'].str.contains('adobe', na=False, regex=False), 'adobe'),
    (df_unique_profiles['company'].str.contains('netflix', na=False, regex=False), 'netflix'),
    (df_unique_profiles['company'].str.contains('starburst', na=False, regex=False), 'starburst'),
    (df_unique_profiles['company'].str.contains('aws', na=False, regex=False), 'amazon'),
    (df_unique_profiles['company'].str.contains('amazon', na=False, regex=False), 'amazon'),
    (df_unique_profiles['email'].str.contains('@databricks', na=False, regex=False), 'databricks'),
    (df_unique_profiles['email'].str.contains('@linkedin', na=False, regex=False), 'linkedin'),
    (df_unique_profiles['email'].str.contains('@amazon', na=False, regex=False), 'amazon')
]

## Update create company_clean column using clean values
company_criteria, company_values = zip(*company_patterns)
df_unique_profiles['company_clean'] = np.select(company_criteria, company_values, None)
df_unique_profiles['company_clean'] = df_unique_profiles['company_clean'].combine_first(df_unique_profiles['company'])

df_unique_profiles['company_clean'] = df_unique_profiles['company_clean'].replace({'@': ''}, regex=True)

# Join dataframes for contributions, pulls, commits to dataframe of unique profiles containing company info
df_all_proj_contributor_profiles = pd.merge(df_unique_profiles, df_all_proj_contributors, on='login', how='outer')
df_pulls_profiles = pd.merge(df_unique_profiles[['company_clean','login']], df_pulls, on='login', how='outer')
df_commits_profiles = pd.merge(df_unique_profiles[['company_clean','login']], df_commits, on='login', how='outer')

# From joined dataframe, create dataframe for unique contributors by company 
df_contributors_by_company = df_all_proj_contributor_profiles.groupby(['project','company_clean'], as_index=False)['login'].count().rename(columns={'login':'contributors'})
df_contributors_by_company['pct_contributors'] = df_contributors_by_company['contributors'] / df_contributors_by_company.groupby('project')['contributors'].transform('sum')

#... and total contributions by company
df_contributions_by_company = df_all_proj_contributor_profiles.groupby(['project','company_clean'], as_index=False)['contributions'].sum().rename(columns={'sum':'contributions'})
df_contributions_by_company['pct_contributions'] = df_contributions_by_company['contributions'] / df_contributions_by_company.groupby('project')['contributions'].transform('sum')

#... and total pull requests by company
df_pulls_by_company = df_pulls_profiles[['project','company_clean']].groupby(['project','company_clean'], as_index=False).size().rename(columns={'size':'pulls'})
df_pulls_by_company['pct_pulls'] = df_pulls_by_company['pulls'] / df_pulls_by_company.groupby('project')['pulls'].transform('sum')

#... and total commits by company
df_commits_by_company = df_commits_profiles[['project','company_clean']].groupby(['project','company_clean'], as_index=False).size().rename(columns={'size':'commits'})
df_commits_by_company['pct_commits'] = df_commits_by_company['commits'] / df_commits_by_company.groupby('project')['commits'].transform('sum')

# DEFINE CHART DATAFRAMES - CONTRIBUTIONS
df_total_contributions = df_all_proj_contributors.groupby('project', as_index=False)['contributions'].sum()

# DEFINE CHART DATAFRAMES - CONTRIBUTIONS - ICEBERG
df_contributions_by_company_iceberg = df_contributions_by_company.loc[df_contributions_by_company['project'] == 'iceberg']
df_contributions_by_company_iceberg = df_contributions_by_company_iceberg.sort_values('contributions', ascending=False)
df_contributions_by_company_iceberg_small = df_contributions_by_company_iceberg.iloc[:5]
df_contributions_by_company_iceberg_small = df_contributions_by_company_iceberg_small.append(
    {
        'project':'iceberg',
        'company_clean':'others',
        'contributions':df_contributions_by_company_iceberg['contributions'].iloc[5:].sum(),
        'pct_contributions':df_contributions_by_company_iceberg['pct_contributions'].iloc[5:].sum()
    }, ignore_index=True
)

# DEFINE CHART DATAFRAMES - CONTRIBUTIONS - DELTA
df_contributions_by_company_delta = df_contributions_by_company.loc[df_contributions_by_company['project'] == 'delta']
df_contributions_by_company_delta = df_contributions_by_company_delta.sort_values('contributions', ascending=False)
df_contributions_by_company_delta_small = df_contributions_by_company_delta.iloc[:5]
df_contributions_by_company_delta_small = df_contributions_by_company_delta_small.append(
    {
        'project':'delta',
        'company_clean':'others',
        'contributions':df_contributions_by_company_delta['contributions'].iloc[5:].sum(),
        'pct_contributions':df_contributions_by_company_delta['pct_contributions'].iloc[5:].sum()
    }, ignore_index=True
)

# DEFINE CHART DATAFRAMES - CONTRIBUTIONS - HUDI
df_contributions_by_company_hudi = df_contributions_by_company.loc[df_contributions_by_company['project'] == 'hudi']
df_contributions_by_company_hudi = df_contributions_by_company_hudi.sort_values('contributions', ascending=False)
df_contributions_by_company_hudi_small = df_contributions_by_company_hudi.iloc[:5]
df_contributions_by_company_hudi_small = df_contributions_by_company_hudi_small.append(
    {
        'project':'hudi',
        'company_clean':'others',
        'contributions':df_contributions_by_company_hudi['contributions'].iloc[5:].sum(),
        'pct_contributions':df_contributions_by_company_hudi['pct_contributions'].iloc[5:].sum()
    }, ignore_index=True
)

# DEFINE CHART DATAFRAMES - CONTRIBUTORS
df_total_contributors = df_all_proj_contributors.groupby('project', as_index=False)['login'].count().rename(columns={'login':'contributors'})

# DEFINE CHART DATAFRAMES - CONTRIBUTORS - ICEBERG
df_contributors_by_company_iceberg = df_contributors_by_company.loc[df_contributions_by_company['project'] == 'iceberg']
df_contributors_by_company_iceberg = df_contributors_by_company_iceberg.sort_values('contributors', ascending=False)
df_contributors_by_company_iceberg_small = df_contributors_by_company_iceberg.iloc[:5]
df_contributors_by_company_iceberg_small = df_contributors_by_company_iceberg_small.append(
    {
        'project':'iceberg',
        'company_clean':'others',
        'contributors':df_contributors_by_company_iceberg['contributors'].iloc[5:].sum(),
        'pct_contributors':df_contributors_by_company_iceberg['pct_contributors'].iloc[5:].sum()
    }, ignore_index=True
)

# DEFINE CHART DATAFRAMES - CONTRIBUTORS - DELTA
df_contributors_by_company_delta = df_contributors_by_company.loc[df_contributions_by_company['project'] == 'delta']
df_contributors_by_company_delta = df_contributors_by_company_delta.sort_values('contributors', ascending=False)
df_contributors_by_company_delta_small = df_contributors_by_company_delta.iloc[:5]
df_contributors_by_company_delta_small = df_contributors_by_company_delta_small.append(
    {
        'project':'delta',
        'company_clean':'others',
        'contributors':df_contributors_by_company_delta['contributors'].iloc[5:].sum(),
        'pct_contributors':df_contributors_by_company_delta['pct_contributors'].iloc[5:].sum()
    }, ignore_index=True
)

# DEFINE CHART DATAFRAMES - CONTRIBUTORS - HUDI
df_contributors_by_company_hudi = df_contributors_by_company.loc[df_contributions_by_company['project'] == 'hudi']
df_contributors_by_company_hudi = df_contributors_by_company_hudi.sort_values('contributors', ascending=False)
df_contributors_by_company_hudi_small = df_contributors_by_company_hudi.iloc[:5]
df_contributors_by_company_hudi_small = df_contributors_by_company_hudi_small.append(
    {
        'project':'hudi',
        'company_clean':'others',
        'contributors':df_contributors_by_company_hudi['contributors'].iloc[5:].sum(),
        'pct_contributors':df_contributors_by_company_hudi['pct_contributors'].iloc[5:].sum()
    }, ignore_index=True
)

df_contributors_by_company_small_xaxis = [
            df_contributors_by_company_iceberg_small['contributors'].max(),
            df_contributors_by_company_delta_small['contributors'].max(),
            df_contributors_by_company_hudi_small['contributors'].max(),
]

# DEFINE CHART DATAFRAMES - PULLS - BARS
df_total_pulls_bars = df_pulls_profiles.groupby('project', as_index=False).size().rename(columns={'size':'pulls'})

# DEFINE CHART DATAFRAMES - PULLS - LINES
df_pulls_profiles['created_at'] = pd.to_datetime(df_pulls_profiles['created_at']).dt.strftime('%Y-%m')
df_total_pulls_line_project = df_pulls_profiles.groupby(['project', 'created_at'], as_index=False).size().rename(columns={'size':'pulls'})
df_total_pulls_line_project['cumsum_pulls'] = df_total_pulls_line_project.groupby(['project'], as_index=False).cumsum()
df_total_pulls_line_company_project = df_pulls_profiles.groupby(['project', 'company_clean', 'created_at'], as_index=False).size().rename(columns={'size':'pulls'})
df_total_pulls_line_company_project['cumsum_pulls'] = df_total_pulls_line_company_project.groupby(['project', 'company_clean'], as_index=False).cumsum()

# DEFINE CHART DATAFRAMES - PULLS - ICEBERG
df_pulls_by_company_iceberg = df_pulls_by_company.loc[df_pulls_by_company['project'] == 'iceberg']
df_pulls_by_company_iceberg = df_pulls_by_company_iceberg.sort_values('pulls', ascending=False)
df_pulls_by_company_iceberg_small = df_pulls_by_company_iceberg.iloc[:5]
df_pulls_by_company_iceberg_small = df_pulls_by_company_iceberg_small.append(
    {
        'project':'iceberg',
        'company_clean':'others',
        'pulls':df_pulls_by_company_iceberg['pulls'].iloc[5:].sum(),
        'pct_pulls':df_pulls_by_company_iceberg['pct_pulls'].iloc[5:].sum()
    }, ignore_index=True
)
df_pulls_by_company_iceberg_line = df_total_pulls_line_company_project.loc[df_total_pulls_line_company_project['project'] == 'iceberg']

# DEFINE CHART DATAFRAMES - PULLS - DELTA
df_pulls_by_company_delta = df_pulls_by_company.loc[df_pulls_by_company['project'] == 'delta']
df_pulls_by_company_delta = df_pulls_by_company_delta.sort_values('pulls', ascending=False)
df_pulls_by_company_delta_small = df_pulls_by_company_delta.iloc[:5]
df_pulls_by_company_delta_small = df_pulls_by_company_delta_small.append(
    {
        'project':'delta',
        'company_clean':'others',
        'pulls':df_pulls_by_company_delta['pulls'].iloc[5:].sum(),
        'pct_pulls':df_pulls_by_company_delta['pct_pulls'].iloc[5:].sum()
    }, ignore_index=True
)
df_pulls_by_company_delta_line = df_total_pulls_line_company_project.loc[df_total_pulls_line_company_project['project'] == 'delta']

# DEFINE CHART DATAFRAMES - PULLS - HUDI
df_pulls_by_company_hudi = df_pulls_by_company.loc[df_pulls_by_company['project'] == 'hudi']
df_pulls_by_company_hudi = df_pulls_by_company_hudi.sort_values('pulls', ascending=False)
df_pulls_by_company_hudi_small = df_pulls_by_company_hudi.iloc[:5]
df_pulls_by_company_hudi_small = df_pulls_by_company_hudi_small.append(
    {
        'project':'hudi',
        'company_clean':'others',
        'pulls':df_pulls_by_company_hudi['pulls'].iloc[5:].sum(),
        'pct_pulls':df_pulls_by_company_hudi['pct_pulls'].iloc[5:].sum()
    }, ignore_index=True
)
df_pulls_by_company_hudi_line = df_total_pulls_line_company_project.loc[df_total_pulls_line_company_project['project'] == 'hudi']

# DEFINE CHART DATAFRAMES - COMMITS - BARS
df_total_commits = df_commits_profiles.groupby('project', as_index=False).size().rename(columns={'size':'commits'})

# DEFINE CHART DATAFRAMES - COMMITS - LINES
df_commits_profiles['created_at'] = pd.to_datetime(df_commits_profiles['created_at']).dt.strftime('%Y-%m')
df_total_commits_line_project = df_commits_profiles.groupby(['project', 'created_at'], as_index=False).size().rename(columns={'size':'commits'})
df_total_commits_line_project['cumsum_commits'] = df_total_commits_line_project.groupby(['project'], as_index=False).cumsum()
df_total_commits_line_company_project = df_commits_profiles.groupby(['project', 'company_clean', 'created_at'], as_index=False).size().rename(columns={'size':'commits'})
df_total_commits_line_company_project['cumsum_commits'] = df_total_commits_line_company_project.groupby(['company_clean'], as_index=False).cumsum()

# DEFINE CHART DATAFRAMES - COMMITS - ICEBERG
df_commits_by_company_iceberg = df_commits_by_company.loc[df_commits_by_company['project'] == 'iceberg']
df_commits_by_company_iceberg = df_commits_by_company_iceberg.sort_values('commits', ascending=False)
df_commits_by_company_iceberg_small = df_commits_by_company_iceberg.iloc[:5]
df_commits_by_company_iceberg_small = df_commits_by_company_iceberg_small.append(
    {
        'project':'iceberg',
        'company_clean':'others',
        'commits':df_commits_by_company_iceberg['commits'].iloc[5:].sum(),
        'pct_commits':df_commits_by_company_iceberg['pct_commits'].iloc[5:].sum()
    }, ignore_index=True
)
df_commits_by_company_iceberg_line = df_total_commits_line_company_project.loc[df_total_commits_line_company_project['project'] == 'iceberg']

# DEFINE CHART DATAFRAMES - COMMITS - DELTA
df_commits_by_company_delta = df_commits_by_company.loc[df_commits_by_company['project'] == 'delta']
df_commits_by_company_delta = df_commits_by_company_delta.sort_values('commits', ascending=False)
df_commits_by_company_delta_small = df_commits_by_company_delta.iloc[:5]
df_commits_by_company_delta_small = df_commits_by_company_delta_small.append(
    {
        'project':'delta',
        'company_clean':'others',
        'commits':df_commits_by_company_delta['commits'].iloc[5:].sum(),
        'pct_commits':df_commits_by_company_delta['pct_commits'].iloc[5:].sum()
    }, ignore_index=True
)
df_commits_by_company_delta_line = df_total_commits_line_company_project.loc[df_total_commits_line_company_project['project'] == 'delta']

# DEFINE CHART DATAFRAMES - COMMITS - HUDI
df_commits_by_company_hudi = df_commits_by_company.loc[df_commits_by_company['project'] == 'hudi']
df_commits_by_company_hudi = df_commits_by_company_hudi.sort_values('commits', ascending=False)
df_commits_by_company_hudi_small = df_commits_by_company_hudi.iloc[:5]
df_commits_by_company_hudi_small = df_commits_by_company_hudi_small.append(
    {
        'project':'hudi',
        'company_clean':'others',
        'commits':df_commits_by_company_hudi['commits'].iloc[5:].sum(),
        'pct_commits':df_commits_by_company_hudi['pct_commits'].iloc[5:].sum()
    }, ignore_index=True
)
df_commits_by_company_hudi_line = df_total_commits_line_company_project.loc[df_total_commits_line_company_project['project'] == 'hudi']


# DEFINE CHARTS - CONTRIBUTIONS
contributions_bars = alt.Chart(df_total_contributions).mark_bar().encode(
        x=alt.X('contributions',
            title=str('total contributions'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('project',
            sort='-x',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('contributions', format=',')
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTIONS - ICEBERG - SMALL
contributions_iceberg_bars_small = alt.Chart(df_contributions_by_company_iceberg_small).mark_bar().encode(
        x=alt.X('contributions',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_contributions_by_company['contributions'].max()])
                ),
        y=alt.Y('company_clean',
                sort=None,
                axis=alt.Axis(title=None, ticks=False)
                ),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributions', format=',', title=str('total contributions')),
                alt.Tooltip('pct_contributions', format='.2%', title=str('percent of contributions'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTIONS - DELTA - SMALL
contributions_delta_bars_small = alt.Chart(df_contributions_by_company_delta_small).mark_bar().encode(
        x=alt.X('contributions',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_contributions_by_company['contributions'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributions', format=',', title=str('distinct contributions')),
                alt.Tooltip('pct_contributions', format='.2%', title=str('percent of contributions'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTIONS - HUDI - SMALL
contributions_hudi_bars_small = alt.Chart(df_contributions_by_company_hudi_small).mark_bar().encode(
        x=alt.X('contributions',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_contributions_by_company['contributions'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributions', format=',', title=str('distinct contributions')),
                alt.Tooltip('pct_contributions', format='.2%', title=str('percent of contributions'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTIONS - ICEBERG - LARGE
contributions_iceberg_bars_large = alt.Chart(df_contributions_by_company_iceberg).mark_bar().encode(
        x=alt.X('contributions',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_contributions_by_company['contributions'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributions', format=',', title=str('distinct contributions')),
                alt.Tooltip('pct_contributions', format='.2%', title=str('percent of contributions'))
            ]
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTIONS - DELTA - LARGE
contributions_delta_bars_large = alt.Chart(df_contributions_by_company_delta).mark_bar().encode(
        x=alt.X('contributions',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_contributions_by_company['contributions'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributions', format=',', title=str('distinct contributions')),
                alt.Tooltip('pct_contributions', format='.2%', title=str('percent of contributions'))
            ]
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTIONS - HUDI - LARGE
contributions_hudi_bars_large = alt.Chart(df_contributions_by_company_hudi).mark_bar().encode(
        x=alt.X('contributions',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_contributions_by_company['contributions'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributions', format=',', title=str('distinct contributions')),
                alt.Tooltip('pct_contributions', format='.2%', title=str('percent of contributions'))
            ]
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )


# DEFINE CHARTS - CONTRIBUTORS
contributors_bars = alt.Chart(df_total_contributors).mark_bar().encode(
        x=alt.X('contributors',
            title=str('distinct contributors'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('project',
            sort='-x',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('contributors', format=',')
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTORS - ICEBERG - SMALL
contributors_iceberg_bars_small = alt.Chart(df_contributors_by_company_iceberg_small).mark_bar().encode(
        x=alt.X('contributors',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,max(df_contributors_by_company_small_xaxis)])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributors', format=',', title=str('distinct contributors')),
                alt.Tooltip('pct_contributors', format='.2%', title=str('percent of contributors'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTORS - DELTA - SMALL
contributors_delta_bars_small = alt.Chart(df_contributors_by_company_delta_small).mark_bar().encode(
        x=alt.X('contributors',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,max(df_contributors_by_company_small_xaxis)])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributors', format=',', title=str('distinct contributors')),
                alt.Tooltip('pct_contributors', format='.2%', title=str('percent of contributors'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - CONTRIBUTORS - HUDI - SMALL
contributors_hudi_bars_small = alt.Chart(df_contributors_by_company_hudi_small).mark_bar().encode(
        x=alt.X('contributors',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,max(df_contributors_by_company_small_xaxis)])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('contributors', format=',', title=str('distinct contributors')),
                alt.Tooltip('pct_contributors', format='.2%', title=str('percent of contributors'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - BARS
pulls_bars = alt.Chart(df_total_pulls_bars).mark_bar().encode(
        x=alt.X('pulls',
            title=str('total pull requests'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('project',
            sort='-x',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('pulls', format=',', title=str('total pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - LINE
pulls_line_project = alt.Chart(df_total_pulls_line_project).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=True)
            ),
        y=alt.Y('pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            alt.Tooltip('pulls', format=',', title=str('total pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

pulls_line_project_cumsum = alt.Chart(df_total_pulls_line_project).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('cumsum_pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            alt.Tooltip('cumsum_pulls', format=',', title=str('cumulative pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - ICEBERG - SMALL
pulls_iceberg_bars_small = alt.Chart(df_pulls_by_company_iceberg_small).mark_bar().encode(
        x=alt.X('pulls',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_pulls_by_company['pulls'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('pulls', format=',', title=str('total pull requests')),
                alt.Tooltip('pct_pulls', format='.2%', title=str('percent of pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - ICEBERG - LINE
pulls_iceberg_line_company = alt.Chart(df_pulls_by_company_iceberg_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('pulls', format=',', title=str('total pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

pulls_iceberg_line_company_cumsum = alt.Chart(df_pulls_by_company_iceberg_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'cumsum_pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('cumsum_pulls', format=',', title=str('cumulative pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )


# DEFINE CHARTS - PULLS - DELTA - SMALL
pulls_delta_bars_small = alt.Chart(df_pulls_by_company_delta_small).mark_bar().encode(
        x=alt.X('pulls',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_pulls_by_company['pulls'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('pulls', format=',', title=str('total pull requests')),
                alt.Tooltip('pct_pulls', format='.2%', title=str('percent of pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - DELTA - LINE
pulls_delta_line_company = alt.Chart(df_pulls_by_company_delta_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('pulls', format=',', title=str('total pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

pulls_delta_line_company_cumsum = alt.Chart(df_pulls_by_company_delta_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'cumsum_pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('cumsum_pulls', format=',', title=str('cumulative pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - HUDI - SMALL
pulls_hudi_bars_small = alt.Chart(df_pulls_by_company_hudi_small).mark_bar().encode(
        x=alt.X('pulls',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_pulls_by_company['pulls'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('pulls', format=',', title=str('total pull requests')),
                alt.Tooltip('pct_pulls', format='.2%', title=str('percent of pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - PULLS - HUDI - LINE
pulls_hudi_line_company = alt.Chart(df_pulls_by_company_hudi_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('pulls', format=',', title=str('total pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

pulls_hudi_line_company_cumsum = alt.Chart(df_pulls_by_company_hudi_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'cumsum_pulls',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('cumsum_pulls', format=',', title=str('cumulative pull requests'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - COMMITS - BARS
commits_bars = alt.Chart(df_total_commits).mark_bar().encode(
        x=alt.X('commits',
            title=str('total commits'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('project',
            sort='-x',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('commits', format=',', title=str('total commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - COMMITS - ICEBERG
commits_iceberg_bars_small = alt.Chart(df_commits_by_company_iceberg_small).mark_bar().encode(
        x=alt.X('commits',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_commits_by_company['commits'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('commits', format=',', title=str('total commits')),
                alt.Tooltip('pct_commits', format='.2%', title=str('percent of commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_iceberg_line_company = alt.Chart(df_commits_by_company_iceberg_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('commits', format=',', title=str('total commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_iceberg_line_company_cumsum = alt.Chart(df_commits_by_company_iceberg_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'cumsum_commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('cumsum_commits', format=',', title=str('cumulative commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - COMMITS - DELTA
commits_delta_bars_small = alt.Chart(df_commits_by_company_delta_small).mark_bar().encode(
        x=alt.X('commits',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_commits_by_company['commits'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('commits', format=',', title=str('total commits')),
                alt.Tooltip('pct_commits', format='.2%', title=str('percent of commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_delta_line_company = alt.Chart(df_commits_by_company_delta_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('commits', format=',', title=str('total commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_delta_line_company_cumsum = alt.Chart(df_commits_by_company_delta_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'cumsum_commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('cumsum_commits', format=',', title=str('cumulative commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - COMMITS - HUDI
commits_hudi_bars_small = alt.Chart(df_commits_by_company_hudi_small).mark_bar().encode(
        x=alt.X('commits',
                axis=alt.Axis(title=None, ticks=False),
                scale=alt.Scale(domain=[0,df_commits_by_company['commits'].max()])
                ),
        y=alt.Y('company_clean', sort=None, axis=alt.Axis(title=None, ticks=False)),
        tooltip=[
                alt.Tooltip('company_clean', title=str('company')),
                alt.Tooltip('commits', format=',', title=str('total commits')),
                alt.Tooltip('pct_commits', format='.2%', title=str('percent of commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_hudi_line_company = alt.Chart(df_commits_by_company_hudi_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('commits', format=',', title=str('total commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_hudi_line_company_cumsum = alt.Chart(df_commits_by_company_hudi_line).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y(
            'cumsum_commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('company_clean',legend=None),
        tooltip=[
            alt.Tooltip('company_clean', title=str('company')),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            
            alt.Tooltip('cumsum_commits', format=',', title=str('cumulative commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# DEFINE CHARTS - COMMITS - LINES
commits_line_project = alt.Chart(df_total_commits_line_project).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date commit created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            alt.Tooltip('commits', format=',', title=str('total commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

commits_line_project_cumsum = alt.Chart(df_total_commits_line_project).mark_line().encode(
        x=alt.X('created_at:T',
            title=str('date pull request created'),
            axis=alt.Axis(title=None, ticks=False)
            ),
        y=alt.Y('cumsum_commits',
            axis=alt.Axis(title=None, ticks=False)
            ),
        color=alt.Color('project',legend=None),
        tooltip=[
            alt.Tooltip('project'),
            alt.Tooltip('created_at', format='%Y-%M', title=str('month')),
            alt.Tooltip('cumsum_commits', format=',', title=str('cumulative commits'))
            ]
    ).properties(
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )

# BEGIN STREAMLIT COMPONENT LAYOUT
cumsum = st.sidebar.radio(
    "Show running totals for line charts?",
    ("Yes", "No")
)

with tab1:
    st.header('Total Contributions')
    st.markdown("""
    [Contributions](https://docs.github.com/en/get-started/quickstart/github-glossary#contributions) are specific activities on GitHub that include:
    - Committing to a repository's default branch or gh-pages branch
    - Opening an issue
    - Opening a discussion
    - Answering a discussion
    - Proposing a pull request
    - Submitting a pull request review
    - Add activities to a user's timeline on their profile: "Contribution activity"
    """)
    
    tab1_row1_1, tab1_row1_2, tab1_row1_3, tab1_row1_4 = st.columns(4)
    
    with tab1_row1_1:
        st.subheader('By Project')
        st.altair_chart(contributions_bars, use_container_width=True)

    with tab1_row1_2:
        st.subheader('Iceberg')
        st.altair_chart(contributions_iceberg_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_contributions_by_company_iceberg[['company_clean','contributions','pct_contributions']].rename(columns={'company_clean':'company','pct_contributions':'%'}))

    with tab1_row1_3:
        st.subheader('Delta Lake')
        st.altair_chart(contributions_delta_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_contributions_by_company_delta[['company_clean','contributions','pct_contributions']].rename(columns={'company_clean':'company','pct_contributions':'%'}))

    with tab1_row1_4:
        st.subheader('Hudi')
        st.altair_chart(contributions_hudi_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_contributions_by_company_hudi[['company_clean','contributions','pct_contributions']].rename(columns={'company_clean':'company','pct_contributions':'%'}))
    
    st.header('Distinct Contributors')
    st.markdown("""
    A [contributor](https://docs.github.com/en/get-started/quickstart/github-glossary#contributor) is someone who does not have collaborator access to a repository but has contributed to a project and had a pull request they opened merged into the repository.
    """)
    
    tab1_row2_1, tab1_row2_2, tab1_row2_3, tab1_row2_4 = st.columns(4)

    with tab1_row2_1:
        st.subheader('By Project')
        st.altair_chart(contributors_bars, use_container_width=True)

    with tab1_row2_2:
        st.subheader('Iceberg')
        st.altair_chart(contributors_iceberg_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_contributors_by_company_iceberg[['company_clean','contributors','pct_contributors']].rename(columns={'company_clean':'company','pct_contributors':'%'}))

    with tab1_row2_3:
        st.subheader('Delta Lake')
        st.altair_chart(contributors_delta_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_contributors_by_company_delta[['company_clean','contributors','pct_contributors']].rename(columns={'company_clean':'company','pct_contributors':'%'}))

    with tab1_row2_4:
        st.subheader('Hudi')
        st.altair_chart(contributors_hudi_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_contributors_by_company_hudi[['company_clean','contributors','pct_contributors']].rename(columns={'company_clean':'company','pct_contributors':'%'}))
        
with tab2:
    st.header('Total Pull Requests by Project')
    st.markdown("""
    [Pull requests](https://docs.github.com/en/get-started/quickstart/github-glossary#pull-request) are proposed changes to a repository submitted by a user and accepted or rejected by a repository's collaborators. Like issues, pull requests each have their own discussion forum.
    """)
    
    tab2_row1_1, tab2_row1_2 = st.columns((1, 3))

    with tab2_row1_1:
        st.subheader('By Project')
        st.altair_chart(pulls_bars, use_container_width=True)

    with tab2_row1_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Pull Requests Over Time by Project')
            st.altair_chart(pulls_line_project_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Requests Over Time by Project')
            st.altair_chart(pulls_line_project, use_container_width=True)

    tab2_row2_1, tab2_row2_2 = st.columns((1, 3))

    with tab2_row2_1:
        st.subheader('Iceberg')
        st.altair_chart(pulls_iceberg_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_pulls_by_company_iceberg[['company_clean','pulls','pct_pulls']].rename(columns={'company_clean':'company','pulls':'pull requests','pct_pulls':'%'}))

    with tab2_row2_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Pull Requests Over Time by Company')
            st.altair_chart(pulls_iceberg_line_company_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Requests Over Time by Company')
            st.altair_chart(pulls_iceberg_line_company, use_container_width=True)


    tab2_row3_1, tab2_row3_2 = st.columns((1, 3))
    
    with tab2_row3_1:
        st.subheader('Delta Lake')
        st.altair_chart(pulls_delta_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_pulls_by_company_delta[['company_clean','pulls','pct_pulls']].rename(columns={'company_clean':'company','pulls':'pull requests','pct_pulls':'%'}))
    
    with tab2_row3_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Pull Requests Over Time by Company')
            st.altair_chart(pulls_delta_line_company_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Requests Over Time by Company')
            st.altair_chart(pulls_delta_line_company, use_container_width=True)
    
    tab2_row4_1, tab2_row4_2 = st.columns((1, 3))
    
    with tab2_row4_1:
        st.subheader('Hudi')
        st.altair_chart(pulls_hudi_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_pulls_by_company_hudi[['company_clean','pulls','pct_pulls']].rename(columns={'company_clean':'company','pulls':'pull requests','pct_pulls':'%'}))

    with tab2_row4_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Pull Requests Over Time by Company')
            st.altair_chart(pulls_hudi_line_company_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Requests Over Time by Company')
            st.altair_chart(pulls_hudi_line_company, use_container_width=True)
        

with tab3:
    st.header('Total Commits by Project')
    st.markdown("""
    A [commit](https://docs.github.com/en/get-started/quickstart/github-glossary) or "revision", is an individual change to a file (or set of files).
    """)
    
    tab3_row1_1, tab3_row1_2 = st.columns((1, 3))

    with tab3_row1_1:
        st.subheader('By Project')
        st.altair_chart(commits_bars, use_container_width=True)
    
    with tab3_row1_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Commits Over Time by Project')
            st.altair_chart(commits_line_project_cumsum, use_container_width=True)
        else:
            st.subheader('Commits Over Time by Project')
            st.altair_chart(commits_line_project, use_container_width=True)


    tab3_row2_1, tab3_row2_2 = st.columns((1, 3))

    with tab3_row2_1:
        st.subheader('Iceberg')
        st.altair_chart(commits_iceberg_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_commits_by_company_hudi[['company_clean','commits','pct_commits']].rename(columns={'company_clean':'company','pct_commits':'%'}))

    with tab3_row2_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Commits Over Time by Company')
            st.altair_chart(commits_iceberg_line_company_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Commits Over Time by Company')
            st.altair_chart(commits_iceberg_line_company, use_container_width=True)


    tab3_row3_1, tab3_row3_2 = st.columns((1, 3))

    with tab3_row3_1:
        st.subheader('Delta Lake')
        st.altair_chart(commits_delta_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_commits_by_company_hudi[['company_clean','commits','pct_commits']].rename(columns={'company_clean':'company','pct_commits':'%'}))

    with tab3_row3_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Commits Over Time by Company')
            st.altair_chart(commits_delta_line_company_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Commits Over Time by Company')
            st.altair_chart(commits_delta_line_company, use_container_width=True)

    tab3_row4_1, tab3_row4_2 = st.columns((1, 3))

    with tab3_row4_1:
        st.subheader('Hudi')
        st.altair_chart(commits_hudi_bars_small, use_container_width=True)
        with st.expander("See all companies"):
            st.table(df_commits_by_company_hudi[['company_clean','commits','pct_commits']].rename(columns={'company_clean':'company','pct_commits':'%'}))

    with tab3_row4_2:
        if cumsum == 'Yes':
            st.subheader('Cumulative Commits Over Time by Company')
            st.altair_chart(commits_hudi_line_company_cumsum, use_container_width=True)
        else:
            st.subheader('Pull Commits Over Time by Company')
            st.altair_chart(commits_hudi_line_company, use_container_width=True)
