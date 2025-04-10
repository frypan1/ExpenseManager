import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from expenses.models import Expense, Category

User = get_user_model()  # Get the custom User model if applicable

class Command(BaseCommand):
    help = "Populate the database with dummy expense data for a specific user"

    def add_arguments(self, parser):
        parser.add_argument('user_id', type=int, help="User ID for whom to generate expenses")

    def handle(self, *args, **options):
        user_id = options['user_id']
        
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"User with ID {user_id} does not exist."))
            return

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
            category, _ = Category.objects.get_or_create(name=category_name, user=user)  # Filter by user

            transactions.append(Expense(
                date=random.choice(date_range),
                category=category,
                amount=round(random.uniform(min_amount, max_amount), 2),
                product_name=f"{category.name} Item {_}",
                user=user  # Ensure expense is linked to the user
            ))

        Expense.objects.bulk_create(transactions, batch_size=500)
        self.stdout.write(self.style.SUCCESS(f"Database populated with 10,000 transactions for user {user.username} (ID: {user.id})."))
