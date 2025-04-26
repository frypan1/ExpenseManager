import random
import numpy as np
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from expenses.models import Expense, Category

User = get_user_model()

class Command(BaseCommand):
    help = "Populate the database with highly realistic dummy expense data for Indian forecasting scenarios"

    def add_arguments(self, parser):
        parser.add_argument('user_id', type=int, help="User ID for whom to generate expenses")

    def handle(self, *args, **options):
        user_id = options['user_id']
        
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"User with ID {user_id} does not exist."))
            return

        # Define expense categories and their behaviors
        categories_behavior = {
            'Clothing': {'base': (100, 300), 'trend': 0.0003},
            'Healthcare': {'base': (50, 150), 'trend': 0.0001},
            'Transportation': {'base': (20, 70), 'trend': 0.00005},
            'Utilities': {'base': (150, 400), 'trend': 0.0001},
            'Dining': {'base': (20, 100), 'trend': 0.00015, 'weekend_boost': 1.3},
            'Entertainment': {'base': (20, 80), 'trend': 0.00015, 'weekend_boost': 1.4},
            'Groceries': {'base': (50, 200), 'trend': 0.0001},
            'Gifts': {'base': (50, 300), 'trend': 0.0002},
        }

        # Define Indian festival seasonality multipliers
        indian_festival_seasonality = {
            1: {'Pongal/Makar Sankranti': 1.2},
            3: {'Holi': 1.3},
            5: {'Eid-ul-Fitr': 1.25},
            8: {'Raksha Bandhan': 1.4},
            9: {'Ganesh Chaturthi': 1.2},
            10: {'Navratri': 1.5, 'Dussehra': 1.6},
            11: {'Diwali': 2.0},
            12: {'Christmas/New Year': 1.3},  # Still minor for general shopping
        }

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 3, 28)
        total_days = (end_date - start_date).days

        transactions = []
        for day_offset in range(total_days):
            current_date = start_date + timedelta(days=day_offset)
            month = current_date.month
            weekday = current_date.weekday()

            for category_name, behavior in categories_behavior.items():
                min_amount, max_amount = behavior['base']
                trend_multiplier = np.exp(behavior['trend'] * day_offset)

                base_amount = np.random.uniform(min_amount, max_amount)

                # Apply weekend multiplier if available
                weekend_multiplier = behavior.get('weekend_boost', 1.0)
                if weekday >= 5:  # Saturday(5), Sunday(6)
                    base_amount *= weekend_multiplier

                # Apply festival multipliers
                festival_multiplier = 1.0
                if month in indian_festival_seasonality:
                    for fest, multiplier in indian_festival_seasonality[month].items():
                        if random.random() < 0.3:  # 30% chance this day feels 'festival-like' during festival month
                            festival_multiplier *= multiplier

                final_amount = base_amount * festival_multiplier * trend_multiplier

                # Add realistic noise
                noise = np.random.uniform(0.85, 1.2)
                final_amount *= noise

                # Outlier big purchases (more common during Diwali, Eid, Holi shopping etc)
                if random.random() < 0.07:  # 7% chance of big-ticket purchases
                    outlier_factor = random.uniform(2.5, 5)
                    final_amount *= outlier_factor

                # 85% probability that an expense happens that day
                if random.random() < 0.85:
                    category, _ = Category.objects.get_or_create(name=category_name, user=user)

                    transactions.append(Expense(
                        date=current_date,
                        category=category,
                        amount=round(final_amount, 2),
                        product_name=f"{category.name} Item {random.randint(1, 10000)}",
                        user=user
                    ))

        Expense.objects.bulk_create(transactions, batch_size=500)
        self.stdout.write(self.style.SUCCESS(f"Generated {len(transactions)} highly realistic Indian expenses for user {user.username} (ID: {user.id})"))

# import random
# import numpy as np
# from datetime import datetime, timedelta
# from django.core.management.base import BaseCommand
# from django.contrib.auth import get_user_model
# from expenses.models import Expense, Category

# User = get_user_model()

# class Command(BaseCommand):
#     help = "Populate the database with highly realistic dummy expense data for Prophet forecasting"

#     def add_arguments(self, parser):
#         parser.add_argument('user_id', type=int, help="User ID for whom to generate expenses")

#     def handle(self, *args, **options):
#         user_id = options['user_id']
        
#         try:
#             user = User.objects.get(id=user_id)
#         except User.DoesNotExist:
#             self.stderr.write(self.style.ERROR(f"User with ID {user_id} does not exist."))
#             return

#         # Define expense categories and their behaviors
#         categories_behavior = {
#             'Clothing': {'base': (100, 300), 'seasonality': {12: 2.0, 6: 1.5}, 'trend': 0.0003, 'holiday_boost': {11: 1.5, 12: 1.8}},
#             'Healthcare': {'base': (50, 150), 'seasonality': {1: 1.5, 2: 1.3, 10: 1.2}, 'trend': 0.0001},
#             'Transportation': {'base': (20, 70), 'seasonality': {}, 'trend': 0.00005, 'holiday_boost': {12: 1.5}},
#             'Utilities': {'base': (150, 400), 'seasonality': {1: 1.3, 7: 1.2}, 'trend': 0.0001},
#             'Dining': {'base': (20, 100), 'seasonality': {}, 'trend': 0.00015, 'weekend_boost': 1.4, 'holiday_boost': {12: 1.2}},
#             'Entertainment': {'base': (20, 80), 'seasonality': {}, 'trend': 0.00015, 'weekend_boost': 1.5, 'holiday_boost': {12: 1.3}},
#             'Groceries': {'base': (50, 200), 'seasonality': {6: 1.2, 11: 1.3}, 'trend': 0.0001, 'holiday_boost': {11: 1.2, 12: 1.4}},
#         }

#         start_date = datetime(2022, 1, 1)
#         end_date = datetime(2025, 3, 28)
#         total_days = (end_date - start_date).days

#         transactions = []
#         for day_offset in range(total_days):
#             current_date = start_date + timedelta(days=day_offset)
#             month = current_date.month
#             weekday = current_date.weekday()

#             for category_name, behavior in categories_behavior.items():
#                 min_amount, max_amount = behavior['base']
#                 seasonal_multiplier = behavior.get('seasonality', {}).get(month, 1.0)
#                 holiday_multiplier = behavior.get('holiday_boost', {}).get(month, 1.0)
#                 weekend_multiplier = behavior.get('weekend_boost', 1.0 if weekday < 5 else behavior.get('weekend_boost', 1.0))
#                 trend_multiplier = np.exp(behavior['trend'] * day_offset)  # Exponential trend to simulate real-world growth

#                 base_amount = np.random.uniform(min_amount, max_amount)
#                 final_amount = base_amount * seasonal_multiplier * holiday_multiplier * weekend_multiplier * trend_multiplier

#                 # Add more noise (90-120%) for added variance
#                 noise = np.random.uniform(0.9, 1.2)
#                 final_amount *= noise

#                 # Occasionally create an outlier (3-5 times larger than normal)
#                 if random.random() < 0.05:  # 5% chance of a large outlier expense
#                     outlier_factor = random.uniform(3, 5)
#                     final_amount *= outlier_factor

#                 # 85% chance to create an expense
#                 if random.random() < 0.85:
#                     category, _ = Category.objects.get_or_create(name=category_name, user=user)

#                     transactions.append(Expense(
#                         date=current_date,
#                         category=category,
#                         amount=round(final_amount, 2),
#                         product_name=f"{category.name} Item {random.randint(1, 10000)}",
#                         user=user
#                     ))

#         # Bulk create expenses for efficiency
#         Expense.objects.bulk_create(transactions, batch_size=500)
#         self.stdout.write(self.style.SUCCESS(f"Generated {len(transactions)} realistic expenses for user {user.username} (ID: {user.id})"))
