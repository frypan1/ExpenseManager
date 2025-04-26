# import pandas as pd
# import numpy as np
# import holidays
# import xgboost as xgb
# from datetime import timedelta
# from django.utils.timezone import now
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from decimal import Decimal  # at top of file
# from .models import Expense, Forecast, Category

# def forecast_user_expenses(user):
#     print(f"🚀 Starting XGBoost forecasting for user: {user.username}")

#     # Clear previous forecasts
#     deleted_count, _ = Forecast.objects.filter(user=user).delete()
#     print(f"🧹 Deleted {deleted_count} previous forecasts.")

#     # Load expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         print("⚠️ No expenses found for this user. Forecasting aborted.")
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

#     indian_holidays = holidays.country_holidays('IN')

#     for category in df['category'].unique():
#         print(f"\n📊 Processing category: {category}")

#         cat_df = df[df['category'] == category][['ds', 'y']].copy()

#         if len(cat_df) < 30:
#             print(f"⚠️ Skipping '{category}' due to insufficient data ({len(cat_df)} records).")
#             continue

#         # Feature Engineering
#         cat_df['day_of_week'] = cat_df['ds'].dt.dayofweek
#         cat_df['is_weekend'] = cat_df['day_of_week'].isin([5, 6]).astype(int)
#         cat_df['month'] = cat_df['ds'].dt.month
#         cat_df['day_of_month'] = cat_df['ds'].dt.day
#         cat_df['is_holiday'] = cat_df['ds'].isin(indian_holidays).astype(int)
#         cat_df['rolling_7d_mean'] = cat_df['y'].rolling(window=7, min_periods=1).mean()
#         cat_df['rolling_30d_sum'] = cat_df['y'].rolling(window=30, min_periods=1).sum()

#         features = ['day_of_week', 'is_weekend', 'month', 'day_of_month', 'is_holiday', 'rolling_7d_mean', 'rolling_30d_sum']

#         X = cat_df[features]
#         y = cat_df['y']

#         # Split data (for internal evaluation)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#         # Train XGBoost
#         model = xgb.XGBRegressor(
#             n_estimators=100,
#             learning_rate=0.01,
#             max_depth=5,
#             subsample=0.7,
#             colsample_bytree=0.7,
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # Evaluate
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         print(f"📈 Performance for '{category}':")
#         print(f"    MAE  : {mae:.2f}")
#         print(f"    RMSE : {rmse:.2f}")
#         print(f"    R²   : {r2:.2f}")

#         # Forecast future dates
#         future_days = 365
#         last_date = cat_df['ds'].max()

#         future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)
#         future_df = pd.DataFrame({'ds': future_dates})

#         future_df['day_of_week'] = future_df['ds'].dt.dayofweek
#         future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
#         future_df['month'] = future_df['ds'].dt.month
#         future_df['day_of_month'] = future_df['ds'].dt.day
#         future_df['is_holiday'] = future_df['ds'].isin(indian_holidays).astype(int)

#         # Fill rolling features with last known values
#         future_df['rolling_7d_mean'] = cat_df['rolling_7d_mean'].iloc[-1]
#         future_df['rolling_30d_sum'] = cat_df['rolling_30d_sum'].iloc[-1]

#         future_X = future_df[features]
#         future_df['yhat'] = model.predict(future_X)

#         # Only future dates after today
#         future_forecast = future_df[future_df['ds'].dt.date > now().date()]

#         # Aggregated predictions
#         timeframes = {
#             'weekly': 7,
#             'monthly': 30,
#             'yearly': 365
#         }

#         for timeframe, days in timeframes.items():
#             end_date = now().date() + timedelta(days=days)
#             mask = (future_df['ds'].dt.date > now().date()) & (future_df['ds'].dt.date <= end_date)
#             period_sum = future_df.loc[mask, 'yhat'].sum()

#             if period_sum <= 0:
#                 print(f"⚠️ Skipping save for '{category}' ({timeframe}) due to non-positive forecast ({period_sum}).")
#                 continue

#             try:
#                 category_obj = Category.objects.get(name=category, user=user)
#                 Forecast.objects.create(
#                     user=user,
#                     category=category_obj,
#                     predicted_amount=Decimal(str(round(period_sum, 2))),  # <-- notice this
#                     prediction_date=end_date,
#                     timeframe=timeframe,
#                 )
#                 print(f"💾 Saved forecast for '{category}' ({timeframe}): {round(period_sum, 2)} by {end_date}.")
#             except Category.DoesNotExist:
#                 print(f"❌ Category '{category}' not found for user '{user.username}'. Skipping save.")
#                 continue

#     print("\n🎯 Forecasting and performance evaluation completed for all categories.")

import pandas as pd
import numpy as np
import holidays
import xgboost as xgb
from datetime import timedelta
from django.utils.timezone import now
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from decimal import Decimal  # at top of file
from .models import Expense, Forecast, Category

def forecast_user_expenses(user):
    print(f"🚀 Starting XGBoost forecasting for user: {user.username}")

    # Clear previous forecasts
    deleted_count, _ = Forecast.objects.filter(user=user).delete()
    print(f"🧹 Deleted {deleted_count} previous forecasts.")

    # Load expenses
    expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
    if not expenses.exists():
        print("⚠️ No expenses found for this user. Forecasting aborted.")
        return

    df = pd.DataFrame(expenses)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

    indian_holidays = holidays.country_holidays('IN')

    for category in df['category'].unique():
        print(f"\n📊 Processing category: {category}")

        cat_df = df[df['category'] == category][['ds', 'y']].copy()

        if len(cat_df) < 30:
            print(f"⚠️ Skipping '{category}' due to insufficient data ({len(cat_df)} records).")
            continue

        # Feature Engineering
        cat_df['day_of_week'] = cat_df['ds'].dt.dayofweek
        cat_df['is_weekend'] = cat_df['day_of_week'].isin([5, 6]).astype(int)
        cat_df['month'] = cat_df['ds'].dt.month
        cat_df['day_of_month'] = cat_df['ds'].dt.day
        cat_df['is_holiday'] = cat_df['ds'].isin(indian_holidays).astype(int)
        cat_df['rolling_7d_mean'] = cat_df['y'].rolling(window=7, min_periods=1).mean()
        cat_df['rolling_30d_sum'] = cat_df['y'].rolling(window=30, min_periods=1).sum()

        features = ['day_of_week', 'is_weekend', 'month', 'day_of_month', 'is_holiday', 'rolling_7d_mean', 'rolling_30d_sum']

        X = cat_df[features]
        y = cat_df['y']

        # Split data (for internal evaluation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"📈 Performance for '{category}':")
        print(f"    MAE  : {mae:.2f}")
        print(f"    RMSE : {rmse:.2f}")
        print(f"    R²   : {r2:.2f}")

        # Forecast future dates
        future_days = 365
        last_date = cat_df['ds'].max()

        future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)
        future_df = pd.DataFrame({'ds': future_dates})

        future_df['day_of_week'] = future_df['ds'].dt.dayofweek
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        future_df['month'] = future_df['ds'].dt.month
        future_df['day_of_month'] = future_df['ds'].dt.day
        future_df['is_holiday'] = future_df['ds'].isin(indian_holidays).astype(int)

        # Fill rolling features with last known values
        future_df['rolling_7d_mean'] = cat_df['rolling_7d_mean'].iloc[-1]
        future_df['rolling_30d_sum'] = cat_df['rolling_30d_sum'].iloc[-1]

        future_X = future_df[features]
        future_df['yhat'] = model.predict(future_X)

        # Only future dates after today
        future_forecast = future_df[future_df['ds'].dt.date > now().date()]

        # Aggregated predictions
        timeframes = {
            'weekly': 7,
            'monthly': 30,
            'yearly': 365
        }

        for timeframe, days in timeframes.items():
            end_date = now().date() + timedelta(days=days)
            mask = (future_df['ds'].dt.date > now().date()) & (future_df['ds'].dt.date <= end_date)
            
            # Ensure the result is Decimal before saving
            period_sum = Decimal(str(future_df.loc[mask, 'yhat'].sum()))

            if period_sum <= 0:
                print(f"⚠️ Skipping save for '{category}' ({timeframe}) due to non-positive forecast ({period_sum}).")
                continue

            try:
                category_obj = Category.objects.get(name=category, user=user)
                Forecast.objects.create(
                    user=user,
                    category=category_obj,
                    predicted_amount=Decimal(str(round(period_sum, 2))),  # Ensure this is Decimal
                    prediction_date=end_date,
                    timeframe=timeframe,
                )
                print(f"💾 Saved forecast for '{category}' ({timeframe}): {round(period_sum, 2)} by {end_date}.")
            except Category.DoesNotExist:
                print(f"❌ Category '{category}' not found for user '{user.username}'. Skipping save.")
                continue

    print("\n🎯 Forecasting and performance evaluation completed for all categories.")


# import pandas as pd
# import numpy as np
# import holidays
# import xgboost as xgb
# from datetime import timedelta
# from django.utils.timezone import now
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from scipy.stats import uniform, randint
# from decimal import Decimal
# from .models import Expense, Forecast, Category

# def forecast_user_expenses(user):
#     print(f"🚀 Starting XGBoost forecasting for user: {user.username}")

#     # Clear previous forecasts
#     deleted_count, _ = Forecast.objects.filter(user=user).delete()
#     print(f"🧹 Deleted {deleted_count} previous forecasts.")

#     # Load expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         print("⚠️ No expenses found for this user. Forecasting aborted.")
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

#     indian_holidays = holidays.country_holidays('IN')

#     for category in df['category'].unique():
#         print(f"\n📊 Processing category: {category}")

#         cat_df = df[df['category'] == category][['ds', 'y']].copy()

#         if len(cat_df) < 30:
#             print(f"⚠️ Skipping '{category}' due to insufficient data ({len(cat_df)} records).")
#             continue

#         # Feature Engineering
#         cat_df['day_of_week'] = cat_df['ds'].dt.dayofweek
#         cat_df['is_weekend'] = cat_df['day_of_week'].isin([5, 6]).astype(int)
#         cat_df['month'] = cat_df['ds'].dt.month
#         cat_df['day_of_month'] = cat_df['ds'].dt.day
#         cat_df['is_holiday'] = cat_df['ds'].isin(indian_holidays).astype(int)
#         cat_df['rolling_7d_mean'] = cat_df['y'].rolling(window=7, min_periods=1).mean()
#         cat_df['rolling_30d_sum'] = cat_df['y'].rolling(window=30, min_periods=1).sum()

#         features = ['day_of_week', 'is_weekend', 'month', 'day_of_month', 'is_holiday', 'rolling_7d_mean', 'rolling_30d_sum']

#         X = cat_df[features]
#         y = cat_df['y']

#         # Split data (for internal evaluation)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#         # Define the hyperparameter search space
#         param_dist = {
#             'n_estimators': randint(100, 200),
#             'learning_rate': uniform(0.001, 0.1),
#             'max_depth': randint(3, 10),
#             'subsample': uniform(0.5, 0.5),
#             'colsample_bytree': uniform(0.5, 0.5),
#             'gamma': uniform(0, 0.5),
#             'min_child_weight': randint(1, 10)
#         }

#         # Initialize XGBRegressor
#         model = xgb.XGBRegressor(random_state=42)

#         # Use RandomizedSearchCV to search for the best hyperparameters
#         randomized_search = RandomizedSearchCV(
#             estimator=model,
#             param_distributions=param_dist,
#             n_iter=20,  # Number of iterations to sample from the parameter space
#             cv=3,       # Cross-validation splits
#             verbose=2,  # Verbosity level
#             random_state=42,
#             n_jobs=-1   # Use all available CPU cores
#         )

#         # Fit the model with RandomizedSearchCV
#         randomized_search.fit(X_train, y_train)

#         # Best model after hyperparameter tuning
#         best_model = randomized_search.best_estimator_

#         # Evaluate the best model
#         y_pred = best_model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         print(f"📈 Performance for '{category}':")
#         print(f"    MAE  : {mae:.2f}")
#         print(f"    RMSE : {rmse:.2f}")
#         print(f"    R²   : {r2:.2f}")

#         # Forecast future dates
#         future_days = 365
#         last_date = cat_df['ds'].max()

#         future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)
#         future_df = pd.DataFrame({'ds': future_dates})

#         future_df['day_of_week'] = future_df['ds'].dt.dayofweek
#         future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
#         future_df['month'] = future_df['ds'].dt.month
#         future_df['day_of_month'] = future_df['ds'].dt.day
#         future_df['is_holiday'] = future_df['ds'].isin(indian_holidays).astype(int)

#         # Fill rolling features with last known values
#         future_df['rolling_7d_mean'] = cat_df['rolling_7d_mean'].iloc[-1]
#         future_df['rolling_30d_sum'] = cat_df['rolling_30d_sum'].iloc[-1]

#         future_X = future_df[features]
#         future_df['yhat'] = best_model.predict(future_X)

#         # Only future dates after today
#         future_forecast = future_df[future_df['ds'].dt.date > now().date()]

#         # Aggregated predictions
#         timeframes = {
#             'weekly': 7,
#             'monthly': 30,
#             'yearly': 365
#         }

#         for timeframe, days in timeframes.items():
#             end_date = now().date() + timedelta(days=days)
#             mask = (future_df['ds'].dt.date > now().date()) & (future_df['ds'].dt.date <= end_date)
#             period_sum = future_df.loc[mask, 'yhat'].sum()

#             if period_sum <= 0:
#                 print(f"⚠️ Skipping save for '{category}' ({timeframe}) due to non-positive forecast ({period_sum}).")
#                 continue

#             try:
#                 category_obj = Category.objects.get(name=category, user=user)
#                 Forecast.objects.create(
#                     user=user,
#                     category=category_obj,
#                     predicted_amount=Decimal(str(round(period_sum, 2))),
#                     prediction_date=end_date,
#                     timeframe=timeframe,
#                 )
#                 print(f"💾 Saved forecast for '{category}' ({timeframe}): {round(period_sum, 2)} by {end_date}.")
#             except Category.DoesNotExist:
#                 print(f"❌ Category '{category}' not found for user '{user.username}'. Skipping save.")
#                 continue

#     print("\n🎯 Forecasting and performance evaluation completed for all categories.")


## ARIMA Forecasting
# import pandas as pd
# import numpy as np
# import holidays
# import statsmodels.api as sm
# from datetime import timedelta
# from django.utils.timezone import now
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from decimal import Decimal
# from .models import Expense, Forecast, Category
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller

# # Function to test if the series is stationary
# def test_stationarity(series):
#     result = adfuller(series)
#     p_value = result[1]
#     return p_value < 0.05  # Data is stationary if p-value < 0.05

# # Forecasting function
# def forecast_user_expenses(user):
#     print(f"🚀 Starting ARIMA forecasting for user: {user.username}")

#     # Clear previous forecasts
#     deleted_count, _ = Forecast.objects.filter(user=user).delete()
#     print(f"🧹 Deleted {deleted_count} previous forecasts.")

#     # Load expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         print("⚠️ No expenses found for this user. Forecasting aborted.")
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

#     df['y'] = pd.to_numeric(df['y'], errors='coerce')  # Convert to numeric, replace non-numeric values with NaN

#     # Drop rows with missing values in the 'y' column
#     df = df.dropna(subset=['y'])

#     indian_holidays = holidays.country_holidays('IN')

#     for category in df['category'].unique():
#         print(f"\n📊 Processing category: {category}")

#         cat_df = df[df['category'] == category][['ds', 'y']].copy()

#         if len(cat_df) < 30:
#             print(f"⚠️ Skipping '{category}' due to insufficient data ({len(cat_df)} records).")
#             continue

#         # Feature Engineering (Add more features)
#         cat_df['day_of_week'] = cat_df['ds'].dt.dayofweek
#         cat_df['is_weekend'] = cat_df['day_of_week'].isin([5, 6]).astype(int)
#         cat_df['month'] = cat_df['ds'].dt.month
#         cat_df['day_of_month'] = cat_df['ds'].dt.day
#         cat_df['is_holiday'] = cat_df['ds'].isin(indian_holidays).astype(int)

#         # Set the 'ds' column as the datetime index
#         cat_df.set_index('ds', inplace=True)

#         # Check if the series is stationary
#         if not test_stationarity(cat_df['y']):
#             print(f"📉 Series is not stationary for '{category}', applying differencing...")
#             cat_df['y_diff'] = cat_df['y'].diff().dropna()
#             cat_df.dropna(inplace=True)  # Drop NaN values that result from differencing

#             if not test_stationarity(cat_df['y_diff']):
#                 print(f"⚠️ Series still not stationary for '{category}' after differencing.")
#                 continue
#             cat_df['y'] = cat_df['y_diff']  # Use differenced data

#         # Train ARIMA Model (tuning the order parameters)
#         model = ARIMA(cat_df['y'], order=(5, 1, 0))  # ARIMA(p, d, q)
#         model_fit = model.fit()

#         # Forecasting future values
#         forecast_steps = 365  # Forecast for 365 days ahead
#         forecast = model_fit.forecast(steps=forecast_steps)

#         # Create future dates dataframe
#         last_date = cat_df.index[-1]  # Use index for the last date
#         future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_steps)
#         future_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

#         # Reverting the differencing if the series was differenced
#         if 'y_diff' in cat_df.columns:
#             future_df['yhat'] = future_df['yhat'].cumsum() + cat_df['y'].iloc[-1]

#         # Performance evaluation (on the training set)
#         y_pred = model_fit.predict(start=cat_df.index[0], end=cat_df.index[-1])
#         y_true = cat_df['y']

#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_true, y_pred)

#         print(f"📈 Performance for '{category}':")
#         print(f"    MAE  : {mae:.2f}")
#         print(f"    MSE  : {mse:.2f}")
#         print(f"    RMSE : {rmse:.2f}")
#         print(f"    R²   : {r2:.2f}")

#         # Aggregated predictions (Weekly, Monthly, and Yearly)
#         timeframes = {
#             'weekly': 7,
#             'monthly': 30,
#             'yearly': 365
#         }

#         for timeframe, days in timeframes.items():
#             end_date = now().date() + timedelta(days=days)
#             mask = (future_df['ds'].dt.date > now().date()) & (future_df['ds'].dt.date <= end_date)
#             period_sum = future_df.loc[mask, 'yhat'].sum()

#             if period_sum <= 0:
#                 print(f"⚠️ Skipping save for '{category}' ({timeframe}) due to non-positive forecast ({period_sum}).")
#                 continue

#             try:
#                 category_obj = Category.objects.get(name=category, user=user)
#                 Forecast.objects.create(
#                     user=user,
#                     category=category_obj,
#                     predicted_amount=Decimal(str(round(period_sum, 2))),
#                     prediction_date=end_date,
#                     timeframe=timeframe,
#                 )
#                 print(f"💾 Saved forecast for '{category}' ({timeframe}): {round(period_sum, 2)} by {end_date}.")
#             except Category.DoesNotExist:
#                 print(f"❌ Category '{category}' not found for user '{user.username}'. Skipping save.")
#                 continue

#     print("\n🎯 Forecasting and performance evaluation completed for all categories.")

##prophet
# import pandas as pd
# import numpy as np
# import holidays
# from datetime import timedelta
# from django.utils.timezone import now
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from decimal import Decimal
# from .models import Expense, Forecast, Category
# from prophet import Prophet
# from statsmodels.tsa.stattools import adfuller  # For stationarity test

# # Function to test if the series is stationary (for ARIMA, but we don't need it for Prophet)
# def test_stationarity(series):
#     result = adfuller(series)
#     p_value = result[1]
#     return p_value < 0.05  # Data is stationary if p-value < 0.05

# # Forecasting function using Prophet
# def forecast_user_expenses(user):
#     print(f"🚀 Starting Prophet forecasting for user: {user.username}")

#     # Clear previous forecasts
#     deleted_count, _ = Forecast.objects.filter(user=user).delete()
#     print(f"🧹 Deleted {deleted_count} previous forecasts.")

#     # Load expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         print("⚠️ No expenses found for this user. Forecasting aborted.")
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

#     # # indian_holidays = holidays.country_holidays('IN')

#     # indian_holidays = holidays.CountryHoliday('IN')
#     # holidays_df = pd.DataFrame(list(indian_holidays.items()), columns=['ds', 'holiday'])
#     # holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

#     for category in df['category'].unique():
#         print(f"\n📊 Processing category: {category}")

#         cat_df = df[df['category'] == category][['ds', 'y']].copy()

#         if len(cat_df) < 30:
#             print(f"⚠️ Skipping '{category}' due to insufficient data ({len(cat_df)} records).")
#             continue

#         # Feature Engineering (Add more features)
#         cat_df['day_of_week'] = cat_df['ds'].dt.dayofweek
#         cat_df['is_weekend'] = cat_df['day_of_week'].isin([5, 6]).astype(int)
#         cat_df['month'] = cat_df['ds'].dt.month
#         cat_df['day_of_month'] = cat_df['ds'].dt.day
#         # cat_df['is_holiday'] = cat_df['ds'].isin(indian_holidays).astype(int)

#         # Initialize and fit Prophet model
#         model = Prophet(
#             # holidays=indian_holidays,
#             daily_seasonality=False,
#             weekly_seasonality=True,
#             yearly_seasonality=True
#         )

#         # Add custom holidays to Prophet (optional)
#         model.add_country_holidays(country_name='IN')

#         # Train the Prophet model
#         model.fit(cat_df)

#         # Forecasting future values
#         future = model.make_future_dataframe(periods=365)  # 365 days forecast
#         forecast = model.predict(future)

#         # Performance evaluation (on the training set)
#         y_pred = forecast['yhat'][:len(cat_df)]
#         y_true = cat_df['y']

#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_true, y_pred)

#         print(f"📈 Performance for '{category}':")
#         print(f"    MAE  : {mae:.2f}")
#         print(f"    MSE  : {mse:.2f}")
#         print(f"    RMSE : {rmse:.2f}")
#         print(f"    R²   : {r2:.2f}")

#         # Aggregated predictions (Weekly, Monthly, and Yearly)
#         timeframes = {
#             'weekly': 7,
#             'monthly': 30,
#             'yearly': 365
#         }

#         for timeframe, days in timeframes.items():
#             end_date = now().date() + timedelta(days=days)
#             mask = (forecast['ds'].dt.date > now().date()) & (forecast['ds'].dt.date <= end_date)
#             period_sum = forecast.loc[mask, 'yhat'].sum()

#             if period_sum <= 0:
#                 print(f"⚠️ Skipping save for '{category}' ({timeframe}) due to non-positive forecast ({period_sum}).")
#                 continue

#             try:
#                 category_obj = Category.objects.get(name=category, user=user)
#                 Forecast.objects.create(
#                     user=user,
#                     category=category_obj,
#                     predicted_amount=Decimal(str(round(period_sum, 2))),
#                     prediction_date=end_date,
#                     timeframe=timeframe,
#                 )
#                 print(f"💾 Saved forecast for '{category}' ({timeframe}): {round(period_sum, 2)} by {end_date}.")
#             except Category.DoesNotExist:
#                 print(f"❌ Category '{category}' not found for user '{user.username}'. Skipping save.")
#                 continue

#     print("\n🎯 Forecasting and performance evaluation completed for all categories.")
# from prophet import Prophet
# import pandas as pd
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from datetime import timedelta
# from .models import Expense, Forecast, Category
# from django.utils.timezone import now
# import numpy as np  # Import numpy here
# from sklearn.preprocessing import StandardScaler
# from prophet.diagnostics import cross_validation, performance_metrics

# from prophet import Prophet
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from datetime import timedelta
# from .models import Expense, Forecast, Category
# from django.utils.timezone import now

# def forecast_user_expenses(user):
#     print(f"🚀 Starting forecasting for user: {user.username}")

#     # Clear previous forecasts
#     deleted_count, _ = Forecast.objects.filter(user=user).delete()
#     print(f"🧹 Deleted {deleted_count} previous forecasts.")

#     # Fetch user expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         print("⚠️ No expenses found for this user. Forecasting aborted.")
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

#     for category in df['category'].unique():
#         print(f"\n📊 Processing category: {category}")

#         cat_df = df[df['category'] == category][['ds', 'y']].copy()
        
#         if len(cat_df) < 15:  # At least 15 entries to train Prophet
#             print(f"⚠️ Skipping '{category}' due to insufficient data ({len(cat_df)} records).")
#             continue

#         # Optional: Apply log transformation if needed for normalization
#         # cat_df['y'] = np.log(cat_df['y'] + 1)

#         # Fit Prophet model
#         model = Prophet(
#             daily_seasonality=True,
#             weekly_seasonality=True,
#             yearly_seasonality=True,
#             seasonality_prior_scale=15.0,  # ⬅️ Stronger belief in seasonal patterns
#             changepoint_prior_scale=0.05,
#         )

#         model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
#         model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
#         model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
#         model.add_seasonality(name='weekly', period=7, fourier_order=3)
#         model.add_seasonality(name='daily', period=1, fourier_order=3)
#         model.fit(cat_df)
#         print(f"✅ Model trained for '{category}'.")

#         # Make future predictions (next 365 days)
#         future = model.make_future_dataframe(periods=365, freq='D')  # Use periods to define the forecast length

#         # Get the forecast
#         forecast = model.predict(future)

#         # Reverse log transformation if it was applied earlier
#         # forecast['yhat'] = np.exp(forecast['yhat']) - 1

#         # 🧮 Performance metrics on training data
#         merged = pd.merge(cat_df, forecast[['ds', 'yhat']], on='ds', how='inner')

#         merged['y'] = merged['y'].astype(float)
#         y_true = merged['y']
#         y_pred = merged['yhat']

#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_true, y_pred)
#         mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#         print(f"📈 Performance for '{category}':")
#         print(f"    MAE   : {mae:.2f}")
#         print(f"    MSE   : {mse:.2f}")
#         print(f"    RMSE  : {rmse:.2f}")
#         print(f"    R²    : {r2:.2f}")
#         print(f"    MAPE  : {mape:.2f}%")

#         # Filter only future dates for prediction (avoid past dates)
#         future_forecast = forecast[forecast['ds'].dt.date > now().date()]

#         # Aggregated predictions
#         timeframes = {
#             'weekly': 7,
#             'monthly': 30,
#             'yearly': 365
#         }

#         for timeframe, days in timeframes.items():
#             end_date = now().date() + timedelta(days=days)
#             mask = (forecast['ds'].dt.date > now().date()) & (forecast['ds'].dt.date <= end_date)
#             period_sum = forecast.loc[mask, 'yhat'].sum()

#             if period_sum <= 0:
#                 print(f"⚠️ Skipping save for '{category}' ({timeframe}) due to non-positive forecast ({period_sum}).")
#                 continue

#             try:
#                 category_obj = Category.objects.get(name=category, user=user)
#                 Forecast.objects.create(
#                     user=user,
#                     category=category_obj,
#                     predicted_amount=round(period_sum, 2),
#                     prediction_date=end_date,
#                     timeframe=timeframe,
#                 )
#                 print(f"💾 Saved forecast for '{category}' ({timeframe}): {round(period_sum, 2)} by {end_date}.")
#             except Category.DoesNotExist:
#                 print(f"❌ Category '{category}' not found for user '{user.username}'. Skipping save.")
#                 continue

#     print("\n🎯 Forecasting and performance evaluation completed for all categories.")