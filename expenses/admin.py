from django.contrib import admin
from .models import Expense

# Register your models here.
@admin.register(Expense)
class ExpenseAdmin(admin.ModelAdmin):
    list_display = ('date', 'category', 'amount', 'product_name')  # Specify which fields to display in the list
    list_filter = ('category',)  # Add filtering by category
    search_fields = ('product_name',)  # Enable search by product_name
    ordering = ('-date',)  # Order by date, with the most recent first

from .models import Category  # Import the Category model

# Register the Category model in the admin interface
admin.site.register(Category)

# Register the Forecast model in the admin interface
from .models import Forecast

@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    list_display = ('user', 'category', 'timeframe', 'predicted_amount', 'prediction_date')
    list_filter = ('timeframe', 'prediction_date', 'category', 'user')
    search_fields = ('user__username', 'category__name')
    ordering = ('-prediction_date',)
