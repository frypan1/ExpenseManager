# # models.py
# from django.db import models

# class Expense(models.Model):
#     date = models.DateField()
#     category = models.CharField(max_length=100)
#     amount = models.DecimalField(max_digits=10, decimal_places=2)
#     product_name = models.CharField(max_length=255)

# class Category(models.Model):
#     name = models.CharField(max_length=100, unique=True)
#     def __str__(self):
#         return self.name

# expenses/models.py
# models.py
'''from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.name

class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link each expense to a user
    date = models.DateField()
    category = models.CharField(max_length=100)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    product_name = models.CharField(max_length=255)

    def __str__(self):
        return f'{self.product_name} - {self.amount}'
'''

from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Category(models.Model):
    name = models.CharField(max_length=100)
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True
    )  # Null for global categories, user-specific otherwise

    class Meta:
        unique_together = ('name', 'user')  # Ensure unique categories per user

    def __str__(self):
        return self.name

class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE)  # Link to Category model
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    product_name = models.CharField(max_length=255)

    def __str__(self):
        return f'{self.product_name} - {self.amount}'

# Signal to create default categories for new users
@receiver(post_save, sender=User)
def create_default_categories(sender, instance, created, **kwargs):
    if created:
        default_categories = ['Food', 'Transport', 'Utilities', 'Entertainment']
        for category_name in default_categories:
            Category.objects.create(name=category_name, user=instance)  # Create categories for the new user
