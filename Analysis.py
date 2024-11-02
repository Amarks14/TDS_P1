import pandas as pd
from scipy.stats import pearsonr, linregress
import numpy as np
from datetime import datetime
from collections import Counter


# Loading the datasets :
users_df = pd.read_csv("users.csv")
repos_df = pd.read_csv("repositories.csv")

# Converting 'created_at' columns to datetime:
users_df['created_at'] = pd.to_datetime(users_df['created_at'])
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], utc=True)

# Define a function to output each answer to a file:
def write_answer(index, answer):
    with open("answers.txt", "a") as f:
        f.write(f"{index}. {answer}\n")

# Question 1:- Top 5 users by followers:
top_5_followers = users_df.nlargest(5, 'followers')['login'].tolist()
write_answer(1, ','.join(top_5_followers))

# Question 2:- 5 earliest registered GitHub users:
earliest_5_users = users_df.nsmallest(5, 'created_at')['login'].tolist()
write_answer(2, ','.join(earliest_5_users))

# Question 3:- Top 3 licenses by popularity:
top_licenses = repos_df['license_name'].dropna().value_counts().nlargest(3).index.tolist()
write_answer(3, ','.join(top_licenses))

# Question 4:- Company with most developers:
users_df['company'] = users_df['company'].str.strip().str.lstrip('@').str.upper()
top_company = users_df['company'].value_counts().idxmax()
write_answer(4, top_company)

# Question 5:- Most popular programming language:
top_language = repos_df['language'].mode().iloc[0]
write_answer(5, top_language)

# Question 6:- Second most popular language for users joined after 2020:
users_post_2020 = users_df[users_df['created_at'] > "2020-01-01"]
top_languages_post_2020 = repos_df[repos_df['login'].isin(users_post_2020['login'])]['language'].value_counts()
second_popular_language = top_languages_post_2020.index[1] if len(top_languages_post_2020) > 1 else "N/A"
write_answer(6, second_popular_language)

# Question 7:- Language with the highest average stars per repository:
average_stars_per_language = repos_df.groupby('language')['stargazers_count'].mean().idxmax()
write_answer(7, average_stars_per_language)

# Question 8:- Top 5 users by leader_strength:
users_df['leader_strength'] = users_df['followers'] / (1 + users_df['following'])
top_5_leaders = users_df.nlargest(5, 'leader_strength')['login'].tolist()
write_answer(8, ','.join(top_5_leaders))

# Question 9:- Correlation between followers and public repos:
followers_repos_corr, _ = pearsonr(users_df['followers'], users_df['public_repos'])
write_answer(9, f"{followers_repos_corr:.3f}")

# Question 10:- Regression slope of followers on public repos:
slope_followers_repos, _, _, _, _ = linregress(users_df['public_repos'], users_df['followers'])
write_answer(10, f"{slope_followers_repos:.3f}")

# Question 11:- Correlation between projects and wiki enabled:





# question 12:-




# Question 13:- Regression slope of followers on bio word count:
users_df['bio_word_count'] = users_df['bio'].fillna('').str.split().str.len()
bio_followers_df = users_df[users_df['bio_word_count'] > 0]
slope_bio_followers, _, _, _, _ = linregress(bio_followers_df['bio_word_count'], bio_followers_df['followers'])
write_answer(13, f"{slope_bio_followers:.3f}")

# Question 14:- Top 5 users with the most repositories created on weekends:


# Question 15:- Fraction of users with email when hireable vs not:


# Question 16:- Most common surname:

