import os
from salesRegressor import logger
from typing import Tuple
import pandas as pd
import numpy as np
from salesRegressor.entity.config_entity import DataTransformationConfig


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading sales and store files.")
        sales = pd.read_csv(self.config.sales_file, low_memory=False)
        store = pd.read_csv(self.config.store_file)
        logger.info(f"Sales shape: {sales.shape}; Store shape: {store.shape}")
        return sales, store

    def _clean_sales(self, sales: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning sales data: remove rows with Sales = 0 and trim outliers")
        
        sales = sales.loc[sales["Sales"] > 0].copy()

        sales_99_9_sales = sales["Sales"].quantile(0.999)
        sales_99_9_custs = sales["Customers"].quantile(0.999)

        logger.info(
            f"Sales 99.9 percentile: {sales_99_9_sales:.2f}; Customers 99.9 percentile: {sales_99_9_custs:.2f}"
            )

        sales = sales.loc[sales["Sales"] <= sales_99_9_sales]
        sales = sales.loc[sales["Customers"] <= sales_99_9_custs]

        logger.info(f"Sales shape after cleaning outliers: {sales.shape}")
        return sales

    def _clean_store(self, store: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning store data: fill NaNs for competition and promo columns")
        store = store.copy()

        if "CompetitionDistance" in store.columns:
            median_cd = store["CompetitionDistance"].median()
            store["CompetitionDistance"] = store["CompetitionDistance"].fillna(median_cd)
            logger.info(f"Filled CompetitionDistance NaNs with median: {median_cd:.2f}")

        zero_fill_cols = [
            "CompetitionOpenSinceYear",
            "CompetitionOpenSinceMonth",
            "Promo2SinceYear",
            "Promo2SinceWeek",
            "PromoInterval",
        ]
        for col in zero_fill_cols:
            if col in store.columns:
                store[col] = store[col].fillna(0)
                logger.info(f"Filled {col} NaNs with 0")

        return store

    def _merge(self, sales: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging sales and store on 'Store' with left join.")
        merged = pd.merge(sales, store, on="Store", how="left")
        logger.info(f"Merged shape: {merged.shape}")
        return merged

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Converting Date to datetime and adding Year/Month/Week features.")
        
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        isocal = df["Date"].dt.isocalendar()
        df["Week"] = isocal["week"]
        
        df = df.sort_values(by="Date").reset_index(drop=True)
        return df


    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering (competition duration, expanding aggregations, lags, rolling trend).")
        df = df.copy()

        for col in ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        mask = (df.get("CompetitionOpenSinceYear", 0) != 0) & (df.get("CompetitionOpenSinceMonth", 0) != 0)

        df["CompetitionOpenDuration"] = 0
        if mask.any():
            years_diff = df["Year"].where(mask, 0) - df["CompetitionOpenSinceYear"].where(mask, 0)
            months_diff = df["Month"].where(mask, 0) - df["CompetitionOpenSinceMonth"].where(mask, 0)
            total_months = years_diff * 12 + months_diff
            total_months = total_months.clip(lower=0)
            df.loc[mask, "CompetitionOpenDuration"] = total_months.loc[mask].astype(int)

        logger.info("Computing expanding mean/median of Sales and Customers per Store (shifted by 1).")
        if "Sales" in df.columns:
            df["AvgSalesPerStore"] = (
                df.groupby("Store")["Sales"].expanding()
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            df["MedSalesPerStore"] = (
                df.groupby("Store")["Sales"].expanding()
                .median()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
        if "Customers" in df.columns:
            df["AvgCustomersPerStore"] = (
                df.groupby("Store")["Customers"].expanding()
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            df["MedCustomersPerStore"] = (
                df.groupby("Store")["Customers"].expanding()
                .median()
                .shift(1)
                .reset_index(level=0, drop=True)
            )

        logger.info("Creating lag features for Sales and Customers (lags 1,2,7).")
        df["LastDaySalesPerStore"] = df.groupby("Store")["Sales"].shift(1)
        df["Last2DaysSalesPerStore"] = df.groupby("Store")["Sales"].shift(2)
        df["LastDayCustomersPerStore"] = df.groupby("Store")["Customers"].shift(1)
        df["Last2DaysCustomersPerStore"] = df.groupby("Store")["Customers"].shift(2)
        df["LastWeekSalesPerStore"] = df.groupby("Store")["Sales"].shift(7)
        df["LastWeekCustomersPerStore"] = df.groupby("Store")["Customers"].shift(7)

        if {"Sales", "Customers"}.issubset(df.columns):
            logger.info("Computing Store_AvgCustSpent_Trend (30-day rolling mean of Sales/Customers, shifted by 1).")
            def avg_cust_spent_rolling(g):
                ratio = (g["Sales"] / g["Customers"].replace({0: np.nan}))
                return ratio.rolling(30, min_periods=1).mean().shift(1)

            df["Store_AvgCustSpent_Trend"] = (
                df.groupby("Store")[["Sales", "Customers"]].apply(avg_cust_spent_rolling).reset_index(level=0, drop=True)
            )

        drop_cols = []
        for col in ["Open", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
            if col in df.columns:
                drop_cols.append(col)

        if drop_cols:
            logger.info(f"Dropping columns: {drop_cols}")
            df = df.drop(columns=drop_cols)

        df = df.sort_values("Date").reset_index(drop=True)

        return df

    def _log_transform(self, df: pd.DataFrame, skewed_features=None) -> pd.DataFrame:
        if skewed_features is None:
            skewed_features = ["Sales", "Customers", "CompetitionDistance"]

        df = df.copy()
        to_apply = [c for c in skewed_features if c in df.columns]
        logger.info(f"Applying log1p transform to columns: {to_apply}")
        for c in to_apply:
            df[c] = np.log1p(df[c].astype(float))
        return df

    def _train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Splitting dataset into train/test with test_size={self.config.test_size}")
        
        df = df.sort_values("Date").reset_index(drop=True)
        n_total = len(df)
        split_index = int(n_total * (1 - float(self.config.test_size)))
        logger.info(f"Total rows: {n_total}; split index: {split_index}")
        train_df = df.iloc[:split_index].reset_index(drop=True)
        test_df = df.iloc[split_index:].reset_index(drop=True)
        logger.info(f"Train shape: {train_df.shape}; Test shape: {test_df.shape}")
        
        return train_df, test_df