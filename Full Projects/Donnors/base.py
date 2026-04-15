
# %%
import polars as pl
from pathlib import Path

# %%
DATA_DIR = Path(__file__).resolve().parents[2] / "raw_data" / "fsu_donor_data"

# %%
base = pl.read_parquet(DATA_DIR / "base_table.parquet").drop(
    ["CurrentYearMembershipPledged", "CurrentYearMembershipPaid"]
)

# %%
# Filter to CY pledgers only - data shows virtually nobody pays without pledging first
model_df = base.filter(pl.col("CYAnnualFundPledged") > 0).with_columns(
    (pl.col("CYAnnualFundPaid") > 0).cast(pl.Int8).alias("target")
)

# %%
# Sanity check
print(model_df.shape)
print(model_df.group_by("target").agg(pl.len().alias("count")).sort("target"))
# %%
model_df.head()

def transformer(rawdf):
    df = rawdf
    df = df.select(
        pl.exclude([
            "CYAnnualFundPledged",
            "CYAnnualFundPaid",
            "FanID",
            "AccountID",
            "CRMID",
            "CurrentYearMembershipLevel",
            "CurrentYearOtherPledged",
            "CurrentYearOtherPaid",
            "MostRecentMembershipLevel"
        ])
    )
    df = df.to_dummies(columns=["state","county","Region","AccountType","MembershipLevel","HighestMembershipLevel","LowestMembershipLevel","PriorYearMembershipLevel","IsAlumni","IsSportParticipant"])
    return df


#%%
from sklearn.model_selection import train_test_split
x = model_df.select(pl.exclude("target"))
y = model_df["target"]
x = transformer(x).to_pandas()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#%%
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

model = XGBClassifier()
# param_grid = {
#     "max_depth": [3, 4, 5],
#     "learning_rate": [0.01, 0.05, 0.1, 0.2],
#     "n_estimators": [100, 200, 300, 400, 500],
#     "subsample": [0.8, 1.0],
# }
# random = RandomizedSearchCV(
#     model,
#     param_distributions=param_grid,
#     n_iter=20,
#     cv=5,
#     scoring="r2", # Changed scoring metric to R^2
#     random_state=42
# )
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    )
# random.fit(X_train, y_train)
# print(random.best_params_)

#%%

model.fit(X_train, y_train)

#%%
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, mean_squared_error
import numpy as np
from lets_plot import *
import pandas as pd
LetsPlot.setup_html()
y_preds = model.predict(X_test)

precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Get feature importances
feature_importances = model.feature_importances_
feature_names = list(X_train.columns)

# Create a DataFrame for better visualization
importance_df = pl.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort('importance', descending=True)

# Print feature importances
print(f'Feature Importance')
print(importance_df)

importance_pd = (
    importance_df
    .head(20)
    .to_pandas()
    .sort_values("importance", ascending=True)
)

importance_pd["feature"] = pd.Categorical(
    importance_pd["feature"],
    categories=importance_pd["feature"],
    ordered=True
)

p = (
    ggplot(importance_pd, aes(x="feature", y="importance"))
    + geom_bar(stat="identity")
    + coord_flip()
)

p.show()

# %%
# Reuse DATA_DIR so paths resolve correctly regardless of working directory.
base = pl.read_parquet(DATA_DIR / "base_table.parquet")
demographic = pl.read_parquet(DATA_DIR / "demographic.parquet")
education = pl.read_parquet(DATA_DIR / "education.parquet")
board_members = pl.read_parquet(DATA_DIR / "board_members.parquet")
ten_year_donation = pl.read_parquet(DATA_DIR / "ten_year_donation.parquet")
wealth = pl.read_parquet(DATA_DIR / "wealth.parquet")
# %%
print(f"base columns: {base.columns}")
print(f"demographic columns: {demographic.columns}")
print(f"education columns: {education.columns}")
print(f"board_members columns: {board_members.columns}")
print(f"ten_year_donation columns: {ten_year_donation.columns}")
print(f"wealth columns: {wealth.columns}")
# %%
ten_year_donation.glimpse()

ten_year_donation.group_by('FanID').agg(pl.col('Most Recent Payment/Pledge Date').min().alias('FirstDonationDate')).sort('FirstDonationDate').tail(20).with_columns(
    pl.col('FirstDonationDate').dt.year().alias('FirstDonationYear'),
    pl.col('FirstDonationDate').when(pl.col('FirstDonationDate').dt.year() < 20260, "").otherwise("Post-2000")
)

# %%

# IWave_AnnualCapacity
# IWave_AnnualCapacityTargetProduction
OUTPUT_DIR = Path(__file__).resolve().parent

base.write_csv(OUTPUT_DIR / "base.csv")
demographic.write_csv(OUTPUT_DIR / "demographic.csv")
education.write_csv(OUTPUT_DIR / "education.csv")
board_members.write_csv(OUTPUT_DIR / "board_members.csv")
ten_year_donation.write_csv(OUTPUT_DIR / "ten_year_donation.csv")
wealth.write_csv(OUTPUT_DIR / "wealth.csv")


# %%
## Did pledge? or did not?

### by education status
base_ed = base.join(education, on="FanID", how="left")
base_ed = base_ed.with_columns(
    pl.when(pl.col("CYAnnualFundPledged") > 0)
    .then(0)
    .otherwise(1)
    .alias("DidNotPledge")
)


education_plot = base_ed.group_by("EducationStatus").agg(
    pl.mean("DidNotPledge").alias("NotPledgeRate"),
    pl.count("DidNotPledge").alias("Count")
).sort("NotPledgeRate", descending=True).filter((pl.col("Count") > 100) & (pl.col("EducationStatus") != "null"))


# %%
from lets_plot import *
LetsPlot.setup_html()
p = (
    ggplot(education_plot.to_pandas(), aes(y="EducationStatus", x="NotPledgeRate"))
    + geom_bar(stat="identity")
    + coord_flip()
)
p
# %%
from lets_plot import *
LetsPlot.setup_html(isolated_frame=False)

# %%
DATA_DIR = Path(__file__).resolve().parents[2] / "raw_data" / "fsu_donor_data"

# %%
base = pl.read_parquet(DATA_DIR /"base_table.parquet").drop(
    ["CurrentYearMembershipPledged", "CurrentYearMembershipPaid"]
)

# %%
# Filter to CY pledgers only - data shows virtually nobody pays without pledging first
model_df = base.filter(pl.col("CYAnnualFundPledged") > 0).with_columns(
    (pl.col("CYAnnualFundPaid") > 0).cast(pl.Int8).alias("target_old"),
    (pl.col("CYAnnualFundPaid") == 0).cast(pl.Int8).alias("target"),
)

# %%
# red - #782F40 gold - #CEB888 black - #101820
education_alumni_plot = model_df.join(education, on="FanID", how="left").group_by("EducationStatus").agg(
    pl.mean("target").alias("NotCommittedRate"),
    pl.count("target").alias("Count"),
    ).sort("NotCommittedRate", descending=True).filter((pl.col("EducationStatus") != "Unknown")).with_columns(
        (pl.col("NotCommittedRate") * 100).round(2).cast(pl.Utf8).add("%").alias("NotCommittedRateLabel")
    )
education_alumni_plot
#%%
p = (
    ggplot(education_alumni_plot.to_pandas(), aes(x="EducationStatus", y="NotCommittedRate"))
    + geom_bar(stat="identity", position="dodge")
    # + geom_text(
    #     aes(label='NotCommittedRateLabel'),
    #     position=position_dodge(width=0.9),  # mismo position que geom_bar
    #     vjust=-0.8,                           # arriba de cada barra,
    #     hjust=0.5,
    #     color="black",
    #     size=8
    # )
    + scale_fill_manual(values={'Y': '#782F40', 'N': '#CEB888'})
    + expand_limits(y=0.15)
)
p
# %%
education_alumni_plot = model_df.join(education, on="FanID", how="left").group_by("EducationStatus").agg(
    pl.mean("target").alias("NotCommittedRate"),
    pl.count("target").alias("Count"),
).sort("NotCommittedRate", descending=True).filter((pl.col("EducationStatus") != "Unknown"))

total_count = education_alumni_plot.select(pl.col("Count").sum()).item()
education_alumni_plot = education_alumni_plot.with_columns(
    (pl.col("Count") / total_count).alias("ProportionOfTotal"),
    (pl.col("ProportionOfTotal") * 100).round(2).cast(pl.Utf8).add("%").alias("ProportionLabel")
)
education_alumni_plot
# %%
