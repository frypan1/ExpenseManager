import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from expenses.models import Expense, Category  # Ensure correct app reference

class Command(BaseCommand):
    help = "Populate the database with dummy expense data"

    def handle(self, *args, **kwargs):
        categories_data = {
            'Clothing': (500, 5000),
            'Healthcare': (200, 1000),
            'Transportation': (50, 10000),
            'Utilities': (300, 20000),
            'Dining': (100, 5000),
            'Entertainment': (100, 10000),
            'Groceries': (100, 10000),
        }

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 3, 28)
        date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]

        transactions = []
        for _ in range(10000):
            category_name, (min_amount, max_amount) = random.choice(list(categories_data.items()))
            category, _ = Category.objects.get_or_create(name=category_name)  # Ensure category exists

            transactions.append(Expense(
                date=random.choice(date_range),
                category=category,
                amount=round(random.uniform(min_amount, max_amount), 2),
                product_name=f"{category.name} Item {_}",
            ))

        Expense.objects.bulk_create(transactions, batch_size=500)
        self.stdout.write(self.style.SUCCESS("Database populated with 10,000 transactions."))
