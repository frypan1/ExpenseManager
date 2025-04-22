# # predictor.py
# import pandas as pd
# from datetime import timedelta
# from .models import Expense, Forecast, Category
# from django.utils.timezone import now

# def forecast_user_expenses(user):
#     # Delete previous forecasts
#     Forecast.objects.filter(user=user).delete()

#     # Fetch user expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category'})
    
#     # Resample and predict per category
#     for category in df['category'].unique():
#         cat_df = df[df['category'] == category].set_index('date')
#         for timeframe, rule, days in [('weekly', 'W', 7), ('monthly', 'M', 30), ('yearly', 'Y', 365)]:
#             grouped = cat_df['amount'].resample(rule).sum()
#             if len(grouped) < 2:
#                 continue  # Not enough data to forecast

#             mean = grouped.mean()
#             forecast_date = now().date() + timedelta(days=days)

#             cat_obj = Category.objects.get(name=category, user=user)
#             Forecast.objects.create(
#                 user=user,
#                 category=cat_obj,
#                 predicted_amount=round(mean, 2),
#                 prediction_date=forecast_date,
#                 timeframe=timeframe,
#             )

# from prophet import Prophet
# import pandas as pd
# from datetime import timedelta
# from .models import Expense, Forecast, Category
# from django.utils.timezone import now

# def forecast_user_expenses(user):
#     print(f"Running forecast for {user.username}...")

#     # Clear old forecasts
#     Forecast.objects.filter(user=user).delete()

#     # Fetch user expenses
#     expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
#     if not expenses.exists():
#         print("No expenses found.")
#         return

#     df = pd.DataFrame(expenses)
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

#     for category in df['category'].unique():
#         cat_df = df[df['category'] == category][['ds', 'y']].copy()

#         if len(cat_df) < 10:
#             print(f"Skipping '{category}' — not enough data: {len(cat_df)} records.")
#             continue

#         print(f"Fitting model for category: {category} ({len(cat_df)} rows)")

#         try:
#             model = Prophet(daily_seasonality=True)
#             model.fit(cat_df)
#         except Exception as e:
#             print(f"Error fitting model for {category}: {e}")
#             continue

#         future = model.make_future_dataframe(periods=365, freq='D')  # Extend 1 year for all predictions
#         forecast = model.predict(future)
#         forecast['ds'] = pd.to_datetime(forecast['ds'])
#         forecast['ds_date'] = forecast['ds'].dt.date

#         for timeframe, days in [('weekly', 7), ('monthly', 30), ('yearly', 365)]:
#             target_date = now().date() + timedelta(days=days)
#             row = forecast.loc[forecast['ds_date'] == target_date]

#             if row.empty:
#                 print(f"No forecast found for {category} on {target_date}, trying closest match...")
#                 row = forecast.loc[forecast['ds_date'] >= target_date].head(1)

#             if row.empty:
#                 print(f"Still no forecast for {category} ({timeframe})")
#                 continue

#             predicted = row['yhat'].values[0]

#             try:
#                 cat_obj = Category.objects.get(name=category, user=user)
#             except Category.DoesNotExist:
#                 print(f"Category '{category}' not found for user.")
#                 continue

#             forecast_entry = Forecast.objects.create(
#                 user=user,
#                 category=cat_obj,
#                 predicted_amount=round(predicted, 2),
#                 prediction_date=target_date,
#                 timeframe=timeframe,
#             )

#             print(f"Saved forecast: {category} | {timeframe} | {round(predicted, 2)} on {target_date}")

from prophet import Prophet
import pandas as pd
from datetime import timedelta
from .models import Expense, Forecast, Category
from django.utils.timezone import now

def forecast_user_expenses(user):
    # Clear previous forecasts
    Forecast.objects.filter(user=user).delete()

    # Fetch user expenses
    expenses = Expense.objects.filter(user=user).values('date', 'category__name', 'amount')
    if not expenses.exists():
        return

    df = pd.DataFrame(expenses)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'category__name': 'category', 'date': 'ds', 'amount': 'y'})

    for category in df['category'].unique():
        cat_df = df[df['category'] == category][['ds', 'y']].copy()
        if len(cat_df) < 15:  # At least 15 entries to train Prophet
            continue

        # Fit Prophet model with seasonality
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(cat_df)

        # Forecast next 365 days
        future = model.make_future_dataframe(periods=365, freq='D')
        forecast = model.predict(future)

        # Filter only future dates
        future_forecast = forecast[forecast['ds'].dt.date > now().date()]

        # Aggregated predictions
        timeframes = {
            'weekly': 7,
            'monthly': 30,
            'yearly': 365
        }

        for timeframe, days in timeframes.items():
            end_date = now().date() + timedelta(days=days)
            mask = (forecast['ds'].dt.date > now().date()) & (forecast['ds'].dt.date <= end_date)
            period_sum = forecast.loc[mask, 'yhat'].sum()

            if period_sum <= 0:
                continue  # Avoid saving negative or zero forecasts

            try:
                category_obj = Category.objects.get(name=category, user=user)
                Forecast.objects.create(
                    user=user,
                    category=category_obj,
                    predicted_amount=round(period_sum, 2),
                    prediction_date=end_date,
                    timeframe=timeframe,
                )
            except Category.DoesNotExist:
                continue

    print("✅ Forecasting completed with Prophet.")
