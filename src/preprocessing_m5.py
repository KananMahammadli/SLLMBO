from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

h = 28
max_lags = 57
tr_last = 1913
fday = datetime(2016, 4, 25)
target_col = "sales"
date_col = "date"


# adapted from: https://www.kaggle.com/code/marcogorelli/winning-solution-preprocessing
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


# adapted from: https://www.kaggle.com/code/marcogorelli/winning-solution-preprocess
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how="left")
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


# adapted from: https://www.kaggle.com/code/kneroma/m5-first-public-notebook-under-0-50
def create_df(
    path,
    price_dtypes,
    cal_dtypes,
    sales_cat_cols,
    date_col,
    day_col,
    id_col,
    first_day=1000,
    last_day=1913,
):
    # parse prices and calendar datasets
    df_prices = pd.read_csv(rf"{path}/sell_prices.csv", dtype=price_dtypes)
    for col, col_dtype in price_dtypes.items():
        if col_dtype == "category":
            df_prices[col] = df_prices[col].cat.codes.astype("int16")
            df_prices[col] -= df_prices[col].min()
            df_prices[col] = df_prices[col].astype("category")

    cal = pd.read_csv(rf"{path}/calendar.csv", dtype=cal_dtypes)
    cal[date_col] = pd.to_datetime(cal[date_col])
    for col, col_dtype in cal_dtypes.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
            cal[col] = cal[col].astype("category")

    start_day = first_day
    numcols = [f"{day_col}_{day}" for day in range(start_day, last_day + 1)]

    dtype = {numcol: "float32" for numcol in numcols}
    dtype.update({col: "category" for col in sales_cat_cols})
    df = pd.read_csv(
        rf"{path}/sales_train_validation.csv",
        usecols=sales_cat_cols + numcols,
        dtype=dtype,
    )

    for col in sales_cat_cols:
        if col != id_col:
            df[col] = df[col].cat.codes.astype("int16")
            df[col] -= df[col].min()
        df[col] = df[col].astype("category")

    df = pd.melt(
        df,
        id_vars=sales_cat_cols,
        value_vars=[col for col in df.columns if col.startswith(f"{day_col}_")],
        var_name=day_col,
        value_name="sales",
    )

    df = merge_by_concat(df, cal, [day_col])
    df = merge_by_concat(df, df_prices, ["store_id", "item_id", "wm_yr_wk"])

    df = reduce_mem_usage(df)
    for col in sales_cat_cols:
        df[col] = df[col].astype("category").cat.remove_unused_categories()

    return df


def preprocess(df, id_col, date_col):
    # drop nan values
    df = df.dropna()

    # filter items with insufficient dates
    df = (
        df.groupby(by=id_col, observed=True)
        .filter(lambda x: x[date_col].nunique() > 900)
        .reset_index(drop=True)
    )
    cat_cols = df.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df[col] = df[col].cat.remove_unused_categories()

    return df


# adapted from: https://www.kaggle.com/code/kyakovlev/m5-lags-features
def feature_engineering(df, target_col, id_col, date_col):
    SHIFT_DAY = 28

    LAG_DAYS = [col for col in range(SHIFT_DAY, SHIFT_DAY + 15)]
    df = df.assign(
        **{
            "{}_lag_{}".format(col, lag): df.groupby([id_col], observed=True)[
                col
            ].transform(lambda x: x.shift(lag))
            for lag in LAG_DAYS
            for col in [target_col]
        }
    )

    # Minify lag columns
    for col in list(df):
        if "lag" in col:
            df[col] = df[col].astype(np.float16)

    for i in [7, 14, 30, 60, 180]:
        print("Rolling period:", i)
        df["rolling_mean_" + str(i)] = (
            df.groupby([id_col], observed=True)[target_col]
            .transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean())
            .astype(np.float16)
        )
        df["rolling_std_" + str(i)] = (
            df.groupby([id_col], observed=True)[target_col]
            .transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std())
            .astype(np.float16)
        )

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int8")
        else:
            if date_feat_func == "weekofyear":
                df[date_feat_name] = df[date_col].dt.isocalendar().week.astype("int8")
            else:
                df[date_feat_name] = getattr(df[date_col].dt, date_feat_func).astype(
                    "int8"
                )
    df = df.dropna().reset_index(drop=True)
    cat_cols = df.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df[col] = df[col].cat.remove_unused_categories()

    return df


def select_items(df, target_col, id_col):
    # Calculate the average sales for each id
    avg_sales_per_id = (
        df.groupby(id_col, observed=True)[target_col].mean().reset_index()
    )

    # Initialize KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Fit the model to the data
    kmeans.fit(avg_sales_per_id[[target_col]])

    # Add the cluster labels to the data
    avg_sales_per_id["cluster"] = kmeans.labels_

    # Print the average sales for each cluster
    for cluster in range(3):
        print(
            f"Cluster {cluster} average sales: "
            f"{avg_sales_per_id[avg_sales_per_id['cluster'] == cluster][target_col].mean()}"
        )

    # Select 30 random ids from each cluster
    ids = []
    for cluster in range(3):
        ids.extend(
            avg_sales_per_id[avg_sales_per_id["cluster"] == cluster][id_col]
            .sample(30, random_state=42)
            .tolist()
        )

    # Filter the original dataframe for the selected ids
    df_selected = df[df[id_col].isin(ids)].reset_index(drop=True)

    return df_selected


if __name__ == "__main__":
    path = r"src/data/m5"
    first_day = 1000
    last_day = 1913
    date_col = "date"
    id_col = "id"
    day_col = "d"
    target_col = "sales"

    calendar_types = {
        "event_name_1": "category",
        "event_name_2": "category",
        "event_type_1": "category",
        "event_type_2": "category",
        "weekday": "category",
        "wm_yr_wk": "int16",
        "wday": "int16",
        "month": "int16",
        "year": "int16",
        "snap_CA": "float32",
        "snap_TX": "float32",
        "snap_WI": "float32",
    }

    price_dtypes = {
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int16",
        "sell_price": "float32",
    }

    sales_cat_cols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]

    df = create_df(
        path=path,
        price_dtypes=price_dtypes,
        cal_dtypes=calendar_types,
        sales_cat_cols=sales_cat_cols,
        date_col=date_col,
        id_col=id_col,
        day_col=day_col,
        first_day=first_day,
        last_day=last_day,
    )

    df_preprocessed = preprocess(
        df=df,
        id_col=id_col,
        date_col=date_col,
    )

    df_selected = select_items(
        df=df_preprocessed,
        target_col=target_col,
        id_col=id_col,
    )

    df_fe = feature_engineering(
        df=df_selected,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
    )

    df_fe.to_csv(rf"{path}/m5_parsed.csv", index=False)
