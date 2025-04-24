from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegisterForm, InvoiceUploadForm, ExpenseForm
from .models import Expense, Category, Forecast
from .utils import perform_ocr
from datetime import datetime, timedelta
from django.db.models import Sum
import openai
from django.conf import settings
from django.utils.timezone import now
import json
from collections import defaultdict
from calendar import month_name
from django.template.loader import render_to_string
from django.http import HttpResponse
from io import BytesIO
from xhtml2pdf import pisa
from .predictor import forecast_user_expenses
import logging

# Configure logging for debugging authentication flow
# logger = logging.getLogger(__name__)

openai.api_key = settings.OPENAI_API_KEY

# Registration view
def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created successfully!")
            # logger.info(f"User {user.username} registered and logged in successfully.")
            return redirect('index')
    else:
        form = RegisterForm()
    return render(request, 'registration/register.html', {'form': form})

# Login view
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            # logger.info(f"User {user.username} logged in successfully.")
            return redirect('index')
        else:
            messages.error(request, "Invalid credentials")
            # logger.warning(f"Invalid login attempt for user: {request.POST.get('username')}")
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

# Logout view
def logout_view(request):
    # logger.info(f"User {request.user.username} logged out successfully.")
    logout(request)
    return redirect('login')

@login_required
def index(request):
    selected_year = request.GET.get('year')
    selected_month = request.GET.get('month')
    now_dt = datetime.now()
    year = int(selected_year) if selected_year else now_dt.year
    month = int(selected_month) if selected_month else None

    expenses = Expense.objects.filter(user=request.user)
    if month:
        expenses = expenses.filter(date__year=year, date__month=month)
    else:
        expenses = expenses.filter(date__year=year)

    expenses_by_category = expenses.values('category__name').annotate(total_amount=Sum('amount'))
    category_labels = [e['category__name'] for e in expenses_by_category]
    category_data = [float(e['total_amount']) for e in expenses_by_category]

    monthly_totals = Expense.objects.filter(user=request.user, date__year=year) \
                                    .values('date__month') \
                                    .annotate(total=Sum('amount')) \
                                    .order_by('date__month')
    monthly_labels = [month_name[e['date__month']] for e in monthly_totals]
    monthly_data = [float(e['total']) for e in monthly_totals]

    top_products = expenses.values('product_name').annotate(total=Sum('amount')).order_by('-total')[:5]
    product_labels = [p['product_name'] for p in top_products]
    product_data = [float(p['total']) for p in top_products]

    available_years = Expense.objects.filter(user=request.user).dates('date', 'year')
    available_months = list(enumerate(month_name))[1:]

    extracted_entries = []
    fixed_categories = []
    form = InvoiceUploadForm()

    if request.method == 'POST':
        form = InvoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['image']
            file_path = f'media/uploads/{file.name}'
            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            extracted_text = perform_ocr(file_path)
            extracted_details_text = extract_invoice_details(extracted_text)

            if extracted_details_text:
                for line in extracted_details_text.splitlines():
                    if line.strip():
                        entry = {}
                        for item in line.split(", "):
                            key, value = item.split(": ")
                            entry[key.strip().lower().replace(" ", "_")] = value.strip()
                        extracted_entries.append(entry)

                user_categories = list(Category.objects.filter(user=request.user).values_list('name', flat=True))
                global_categories = list(Category.objects.filter(user__isnull=True).values_list('name', flat=True))
                fixed_categories = user_categories + global_categories

                for entry in extracted_entries:
                    if 'category' in entry and entry['category'] not in fixed_categories:
                        entry['category'] = 'Uncategorized'

                return render(request, 'review.html', {
                    'extracted_entries': extracted_entries,
                    'fixed_categories': fixed_categories,
                })

    # Forecasts
    # forecast_data = {'weekly': [], 'monthly': [], 'yearly': []}
    # forecast_qs = Forecast.objects.filter(user=request.user)

    # for f in forecast_qs:
    #     forecast_data[f.timeframe].append({
    #         'category': f.category.name,
    #         'amount': float(f.predicted_amount),
    #         'date': f.prediction_date.strftime('%Y-%m-%d'),
    #     })

    forecast_user_expenses(request.user)


    forecasts = Forecast.objects.filter(user=request.user)

    # Prepare data for the chart
    weekly_forecasted_data = []
    monthly_forecasted_data = []
    yearly_forecasted_data = []

    for forecast in forecasts:
        if forecast.timeframe == 'weekly':
            weekly_forecasted_data.append({
                'category': forecast.category.name,
                'predicted_amount': float(forecast.predicted_amount),
                'prediction_date': forecast.prediction_date.strftime('%Y-%m-%d'),
            })
        elif forecast.timeframe == 'monthly':
            monthly_forecasted_data.append({
                'category': forecast.category.name,
                'predicted_amount': float(forecast.predicted_amount),
                'prediction_date': forecast.prediction_date.strftime('%Y-%m-%d'),
            })
        elif forecast.timeframe == 'yearly':
            yearly_forecasted_data.append({
                'category': forecast.category.name,
                'predicted_amount': float(forecast.predicted_amount),
                'prediction_date': forecast.prediction_date.strftime('%Y-%m-%d'),
            })

    context = {
        'form': form,
        'category_labels': json.dumps(category_labels),
        'category_data': json.dumps(category_data),
        'monthly_labels': json.dumps(monthly_labels),
        'monthly_data': json.dumps(monthly_data),
        'product_labels': json.dumps(product_labels),
        'product_data': json.dumps(product_data),
        'selected_year': year,
        'selected_month': month,
        'available_years': available_years,
        'available_months': available_months,
        # 'forecast_data': json.dumps(forecast_data), 
        'weekly_forecasted_data': json.dumps(weekly_forecasted_data),
        'monthly_forecasted_data': json.dumps(monthly_forecasted_data),
        'yearly_forecasted_data': json.dumps(yearly_forecasted_data), # <- JSON dump here
    }

    return render(request, 'index.html', context)

def extract_invoice_details(invoice_text):
    try:
        categories = Category.objects.values_list('name', flat=True)
        category_list = ", ".join([f"'{category}'" for category in categories])

        prompt = f"""
        Extract the following details from the invoice:
        1. Date of purchase
        2. Suggested category from this list: [{category_list}]
        3. Amount spent
        4. Product or service name

        Invoice text:
        {invoice_text}

        Return the details in the following format (one entry per line):
        Date: DD-MM-YYYY, Category: <suggested_category>, Amount: <numeric_amount>, Product Name: <product_name>.
        Only return the numeric value for the amount without currency symbols.
        Remove commas from product name.
        """

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error extracting invoice details: {e}")
        return None

@login_required
def confirm(request):
    if request.method == 'POST':
        entry_count = int(request.POST.get('entry_count', 0))
        expense_details_list = []

        for i in range(1, entry_count + 1):
            date_str = request.POST.get(f'date_{i}')
            selected_category = request.POST.get(f'category_{i}')
            custom_category = request.POST.get(f'custom_category_{i}')
            amount = request.POST.get(f'amount_{i}')
            product_name = request.POST.get(f'product_name_{i}')
            category_name = custom_category if selected_category == "Custom" and custom_category else selected_category

            # Create or fetch the category
            category, created = Category.objects.get_or_create(
                name=category_name,
                user=request.user,
            )

            date_obj = datetime.strptime(date_str, '%d-%m-%Y').date()
            expense_details_list.append({
                'date': date_obj,
                'category': category,
                'amount': float(amount),
                'product_name': product_name,
            })

        for details in expense_details_list:
            Expense.objects.create(
                user=request.user,
                date=details['date'],
                category=details['category'],
                amount=details['amount'],
                product_name=details['product_name'],
            )
        print("Expense details saved successfully.")
        # forecast_user_expenses(request.user)
        return redirect('index')

@login_required
def recent_expenses(request):
    today = now().date()
    thirty_days_ago = today - timedelta(days=30)

    recent_expenses = (
        Expense.objects.filter(user=request.user, date__range=[thirty_days_ago, today])
        .order_by('-date')
    )

    return render(request, 'recent_expenses.html', {'recent_expenses': recent_expenses})

# @login_required
# def edit_expense(request, expense_id):
#     expense = get_object_or_404(Expense, id=expense_id, user=request.user)
#     if request.method == 'POST':
#         form = ExpenseForm(request.POST, instance=expense)
#         if form.is_valid():
#             form.save()
#             return redirect('recent_expenses')
#     else:
#         form = ExpenseForm(instance=expense)
#     return render(request, 'edit_expense.html', {'form': form})

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Expense, Category
from .forms import ExpenseForm
from datetime import datetime

@login_required
def edit_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    
    if request.method == 'POST':
        entry_count = int(request.POST.get('entry_count', 1))  # Default to 1 entry if not provided

        for i in range(1, entry_count + 1):
            # Fetch form data for the current entry
            date_str = request.POST.get(f'date_{i}')
            selected_category = request.POST.get(f'category_{i}')
            custom_category = request.POST.get(f'custom_category_{i}')
            amount = request.POST.get(f'amount_{i}')
            product_name = request.POST.get(f'product_name_{i}')

            if not (date_str and amount and product_name and (selected_category or custom_category)):
                continue  # Skip incomplete rows

            # Handle custom category
            category_name = custom_category if selected_category == 'Custom' and custom_category else selected_category

            # Get or create the category for this user
            category, _ = Category.objects.get_or_create(
                name=category_name,
                user=request.user
            )

            # Convert date string to date object
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                continue  # Skip invalid dates

            # Update the expense with new data (or create a new one if needed)
            expense.date = date_obj
            expense.category = category
            expense.amount = amount
            expense.product_name = product_name
            expense.save()

        return redirect('recent_expenses')

    else:
        # On GET request, show the form with current expense data
        form = ExpenseForm(instance=expense)

    # Fetch the user's categories to display them
    categories = Category.objects.filter(user=request.user)
    return render(request, 'edit_expense.html', {
        'form': form,
        'categories': categories,
        'expense': expense,  # For pre-selecting the category in the form
    })


# from django.shortcuts import render, get_object_or_404, redirect
# from django.contrib.auth.decorators import login_required
# from .models import Expense, Category
# from .forms import ExpenseForm

# @login_required
# def edit_expense(request, expense_id):
#     expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    
#     # Fetch the user-specific categories
#     user_categories = Category.objects.filter(user=request.user)
    
#     if request.method == 'POST':
#         form = ExpenseForm(request.POST, instance=expense)
#         # Handling category selection for Custom category
#         selected_category = request.POST.get('category')
#         custom_category = request.POST.get('custom_category')

#         if selected_category == 'Custom' and custom_category:
#             # Create a new custom category if needed
#             category, created = Category.objects.get_or_create(
#                 name=custom_category,
#                 user=request.user
#             )
#             expense.category = category  # Set the new category to the expense
        
#         elif selected_category and selected_category != 'Custom':
#             # If the selected category is not 'Custom', use the selected category
#             category = Category.objects.get(name=selected_category, user=request.user)
#             expense.category = category
        
#         # Save the expense after assigning the correct category
#         if form.is_valid():
#             form.save()
#             return redirect('recent_expenses')
    
#     else:
#         # If GET request, pre-populate the form with the current expense data
#         form = ExpenseForm(instance=expense)

#     # Pass categories and selected category to the template
#     return render(request, 'edit_expense.html', {
#         'form': form,
#         'user_categories': user_categories,
#         'selected_category': expense.category.name,
#     })


@login_required
def delete_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    if request.method == 'POST':
        expense.delete()
        return redirect('recent_expenses')
    return render(request, 'confirm_delete.html', {'expense': expense})

def generate_report(request):
    if request.method == 'POST':
        report_type = request.POST.get('report_type')
        year = request.POST.get('year')
        month = request.POST.get('month')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        category_chart = request.POST.get('category_chart_image')
        monthly_chart = request.POST.get('monthly_chart_image')
        product_chart = request.POST.get('product_chart_image')

        expenses = []
        title = "Expense Report"
        subtitle = ""

        if report_type == 'yearly' and year:
            expenses = Expense.objects.filter(date__year=year)
            subtitle = f"Year: {year}"
        elif report_type == 'monthly' and year and month:
            expenses = Expense.objects.filter(date__year=year, date__month=month)
            subtitle = f"Month: {month}/{year}"
        elif report_type == 'range' and start_date and end_date:
            expenses = Expense.objects.filter(date__range=[start_date, end_date])
            subtitle = f"From {start_date} to {end_date}"

        total_amount = sum(e.amount for e in expenses)
        top_products = Expense.objects.values('product_name').annotate(total_spent=Sum('amount')).order_by('-total_spent')[:5]
        product_labels = [product['product_name'] for product in top_products]
        product_data = [product['total_spent'] for product in top_products]

        html = render_to_string("report_template.html", {
            "expenses": expenses,
            "total": total_amount,
            "title": title,
            "subtitle": subtitle,
            "category_chart": category_chart,
            "monthly_chart": monthly_chart,
            "product_chart": product_chart,
            "product_labels": product_labels,
            "product_data": product_data,
        })

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{title.replace(" ", "_")}.pdf"'
        pisa.CreatePDF(BytesIO(html.encode("UTF-8")), dest=response, encoding='UTF-8')
        return response
    
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from datetime import datetime
from .models import Expense, Category

@login_required
def add_expense(request):
    if request.method == 'POST':
        entry_count = int(request.POST.get('entry_count', 0))

        for i in range(1, entry_count + 1):
            date_str = request.POST.get(f'date_{i}')
            selected_category = request.POST.get(f'category_{i}')
            custom_category = request.POST.get(f'custom_category_{i}')
            amount = request.POST.get(f'amount_{i}')
            product_name = request.POST.get(f'product_name_{i}')

            if not (date_str and amount and product_name and (selected_category or custom_category)):
                continue  # Skip incomplete rows

            # Handle custom category
            category_name = custom_category if selected_category == 'Custom' and custom_category else selected_category

            # Get or create the category for this user
            category, _ = Category.objects.get_or_create(
                name=category_name,
                user=request.user
            )

            # Convert date string to date object
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                continue  # Skip invalid dates

            # Save the expense
            Expense.objects.create(
                user=request.user,
                date=date_obj,
                category=category,
                amount=amount,
                product_name=product_name
            )

        return redirect('index')

    # On GET: show form with user's categories
    categories = Category.objects.filter(user=request.user)
    return render(request, 'add_expense.html', {'categories': categories})
