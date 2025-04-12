from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegisterForm, InvoiceUploadForm, ExpenseForm
from .models import Expense, Category
from .utils import perform_ocr
from datetime import datetime, timedelta
from django.db.models import Sum
import openai
from django.conf import settings
from django.utils.timezone import now

openai.api_key = settings.OPENAI_API_KEY


# Registration view
def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in after successful registration
            messages.success(request, "Account created successfully!")
            return redirect('index')  # Redirect to the index page after registration
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
            return redirect('index')  # Redirect to the index page after login
        else:
            messages.error(request, "Invalid credentials")
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})


# Logout view
def logout_view(request):
    logout(request)
    return redirect('login')  # Redirect to login page after logging out


# Dashboard view (user-specific data)
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.db.models import Sum
from django.db.models.functions import TruncMonth
from .models import Expense, Category
from .forms import InvoiceUploadForm
from .utils import perform_ocr # Assuming you have these

import json
from collections import defaultdict
from datetime import datetime
from calendar import month_name
from .forms import InvoiceUploadForm  # Assuming this exists


@login_required
def index(request):

    #askdbk

    selected_year = request.GET.get('year')
    selected_month = request.GET.get('month')

    now = datetime.now()
    year = int(selected_year) if selected_year else now.year
    month = int(selected_month) if selected_month else None

    expenses = Expense.objects.filter(user=request.user)
    if month:
        expenses = expenses.filter(date__year=year, date__month=month)
    else:
        expenses = expenses.filter(date__year=year)


    expenses_by_category = expenses.values('category__name').annotate(total_amount=Sum('amount'))
    category_labels = [e['category__name'] for e in expenses_by_category]
    category_data = [float(e['total_amount']) for e in expenses_by_category]

    # Line Chart: Monthly totals (only if no month selected)
    monthly_totals = Expense.objects.filter(user=request.user, date__year=year) \
                                    .values('date__month') \
                                    .annotate(total=Sum('amount')) \
                                    .order_by('date__month')
    monthly_labels = [month_name[e['date__month']] for e in monthly_totals]
    monthly_data = [float(e['total']) for e in monthly_totals]

    # Bar Chart: Top 5 products
    top_products = expenses.values('product_name').annotate(total=Sum('amount')).order_by('-total')[:5]
    product_labels = [p['product_name'] for p in top_products]
    product_data = [float(p['total']) for p in top_products]

    available_years = Expense.objects.filter(user=request.user).dates('date', 'year')
    available_months = list(enumerate(month_name))[1:]  # [(1, 'January'), ..., (12, 'December')]

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
    }
            
    #askdjn
    # user = request.user

    # # 1. Category-wise Expenses (Pie Chart)
    # expenses_by_category = (
    #     Expense.objects.filter(user=user)
    #     .values('category__name')
    #     .annotate(total_amount=Sum('amount'))
    #     .order_by('category__name')
    # )
    # category_labels = [e['category__name'] for e in expenses_by_category]
    # category_data = [float(e['total_amount']) for e in expenses_by_category]

    # # 2. Monthly Expenses (Line Chart)
    # expenses_by_month = (
    #     Expense.objects.filter(user=user)
    #     .annotate(month=TruncMonth('date'))
    #     .values('month')
    #     .annotate(total=Sum('amount'))
    #     .order_by('month')
    # )
    # monthly_labels = [e['month'].strftime('%B %Y') for e in expenses_by_month]
    # monthly_data = [float(e['total']) for e in expenses_by_month]

    # # 3. Top Products (Bar Chart)
    # top_products = (
    #     Expense.objects.filter(user=user)
    #     .values('product_name')
    #     .annotate(total=Sum('amount'))
    #     .order_by('-total')[:5]
    # )
    # product_labels = [e['product_name'] for e in top_products]
    # product_data = [float(e['total']) for e in top_products]

    # extracted_entries = []
    # fixed_categories = []
    # form = InvoiceUploadForm()

    # # 4. Invoice Upload + OCR Handling
    # if request.method == 'POST':
    #     form = InvoiceUploadForm(request.POST, request.FILES)
    #     if form.is_valid():
    #         file = form.cleaned_data['image']
    #         file_path = f'media/uploads/{file.name}'
    #         with open(file_path, 'wb') as f:
    #             for chunk in file.chunks():
    #                 f.write(chunk)

    #         extracted_text = perform_ocr(file_path)
    #         extracted_details_text = extract_invoice_details(extracted_text)

    #         if extracted_details_text:
    #             for line in extracted_details_text.splitlines():
    #                 if line.strip():
    #                     entry = {}
    #                     for item in line.split(", "):
    #                         key, value = item.split(": ")
    #                         entry[key.strip().lower().replace(" ", "_")] = value.strip()
    #                     extracted_entries.append(entry)

    #             user_categories = list(Category.objects.filter(user=user).values_list('name', flat=True))
    #             global_categories = list(Category.objects.filter(user__isnull=True).values_list('name', flat=True))
    #             fixed_categories = user_categories + global_categories

    #             for entry in extracted_entries:
    #                 if 'category' in entry and entry['category'] not in fixed_categories:
    #                     entry['category'] = 'Uncategorized'

    #             return render(request, 'review.html', {
    #                 'extracted_entries': extracted_entries,
    #                 'fixed_categories': fixed_categories,
    #             })

    # available_years = Expense.objects.filter(user=request.user).dates('date', 'year')
    # available_months = list(enumerate(month_name))[1:]  # skip empty index


    # # 5. Final Context
    # context = {
    #     'form': form,
    #     'category_labels': json.dumps(category_labels),
    #     'category_data': json.dumps(category_data),
    #     'monthly_labels': json.dumps(monthly_labels),
    #     'monthly_data': json.dumps(monthly_data),
    #     'product_labels': json.dumps(product_labels),
    #     'product_data': json.dumps(product_data),
    # }

    return render(request, 'index.html', context)



# Extract invoice details using OpenAI API
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
        print(response.choices[0].text.strip())
        return response.choices[0].text.strip()

    except Exception as e:
        print(f"Error extracting invoice details: {e}")
        return None


# Confirmation view (saving the extracted details into the database)
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

            # Use custom category if specified
            category_name = custom_category if selected_category == "Custom" and custom_category else selected_category

            # Create or fetch the category
            category, created = Category.objects.get_or_create(
                name=category_name,
                user=request.user,
            )

            date_obj = datetime.strptime(date_str, '%d-%m-%Y').date()

            # Prepare expense details
            expense_details_list.append({
                'date': date_obj,
                'category': category,
                'amount': float(amount),
                'product_name': product_name,
            })

        # Save expenses to the database
        for details in expense_details_list:
            Expense.objects.create(
                user=request.user,
                date=details['date'],
                category=details['category'],
                amount=details['amount'],
                product_name=details['product_name'],
            )

        return redirect('index')


# Recent expenses view
@login_required
def recent_expenses(request):
    today = now().date()
    thirty_days_ago = today - timedelta(days=30)

    recent_expenses = (
        Expense.objects.filter(user=request.user, date__range=[thirty_days_ago, today])
        .order_by('-date')
    )

    return render(request, 'recent_expenses.html', {'recent_expenses': recent_expenses})


# Edit expense view
@login_required
def edit_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    if request.method == 'POST':
        form = ExpenseForm(request.POST, instance=expense)
        if form.is_valid():
            form.save()
            return redirect('recent_expenses')
    else:
        form = ExpenseForm(instance=expense)
    return render(request, 'edit_expense.html', {'form': form})


# Delete expense view
@login_required
def delete_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    if request.method == 'POST':
        expense.delete()
        return redirect('recent_expenses')
    return render(request, 'confirm_delete.html', {'expense': expense})

#hello

# from django.shortcuts import render
# from django.http import HttpResponse
# from django.template.loader import render_to_string
# from weasyprint import HTML
# from datetime import datetime
# from .models import Expense

# # Function to fetch expenses for a specific year
# def fetch_yearly_expenses(year):
#     return Expense.objects.filter(date__year=year)

# # Function to fetch expenses for a specific month of a specific year
# def fetch_monthly_expenses(year, month):
#     return Expense.objects.filter(date__year=year, date__month=month)

# # Function to fetch expenses within a specific date range
# def fetch_expenses_by_date_range(start_date, end_date):
#     start_date = datetime.strptime(start_date, "%Y-%m-%d")
#     end_date = datetime.strptime(end_date, "%Y-%m-%d")
#     return Expense.objects.filter(date__range=[start_date, end_date])

# # Handle report generation
# def generate_report(request):
#     if request.method == 'POST':
#         report_type = request.POST.get('report_type')
#         year = request.POST.get('year')
#         month = request.POST.get('month')
#         start_date = request.POST.get('start_date')
#         end_date = request.POST.get('end_date')

#         # Fetch expenses data based on report type
#         if report_type == 'yearly' and year:
#             expenses = fetch_yearly_expenses(year)
#         elif report_type == 'monthly' and year and month:
#             expenses = fetch_monthly_expenses(year, month)
#         elif report_type == 'daterange' and start_date and end_date:
#             expenses = fetch_expenses_by_date_range(start_date, end_date)
#         else:
#             return HttpResponse("Invalid data received", status=400)

#         # Prepare context for PDF rendering
#         context = {
#             'expenses': expenses,
#             'selected_year': year,
#             'selected_month': month,
#             'start_date': start_date,
#             'end_date': end_date,
#             'report_type': report_type
#         }

#         # Render PDF using WeasyPrint
#         html = render_to_string('report_template.html', context)
#         pdf = HTML(string=html).write_pdf()

#         # Return PDF as a response
#         response = HttpResponse(pdf, content_type='application/pdf')
#         response['Content-Disposition'] = f'attachment; filename="report_{year}_{month if month else ""}_{start_date}_{end_date}.pdf"'
#         return response

#     return HttpResponse(status=405)  # Method not allowed if it's not a POST request
# import base64
# from django.template.loader import render_to_string
# from django.http import HttpResponse
# from xhtml2pdf import pisa
# from io import BytesIO
# from .models import Expense


# def generate_report(request):
#     if request.method == 'POST':
#         report_type = request.POST.get('report_type')
#         year = request.POST.get('year')
#         month = request.POST.get('month')
#         start_date = request.POST.get('start_date')
#         end_date = request.POST.get('end_date')

#         # base64 chart images (URLs like "data:image/png;base64,...")
#         chart_category = request.POST.get('category_chart_image')
#         chart_monthly = request.POST.get('monthly_chart_image')
#         chart_product = request.POST.get('product_chart_image')

#         expenses = []
#         subtitle = ""

#         if report_type == 'yearly' and year:
#             expenses = Expense.objects.filter(date__year=year)
#             subtitle = f"Year: {year}"

#         elif report_type == 'monthly' and year and month:
#             expenses = Expense.objects.filter(date__year=year, date__month=month)
#             subtitle = f"Month: {month}/{year}"

#         elif report_type == 'range' and start_date and end_date:
#             expenses = Expense.objects.filter(date__range=[start_date, end_date])
#             subtitle = f"From {start_date} to {end_date}"

#         total_amount = sum(e.amount for e in expenses)

#         # Render HTML with charts and expense data
#         html = render_to_string("report_template.html", {
#             "report_type": report_type,
#             "selected_year": year,
#             "selected_month": month,
#             "start_date": start_date,
#             "end_date": end_date,
#             "expenses": expenses,
#             "total": total_amount,
#             "chart_category": chart_category,
#             "chart_monthly": chart_monthly,
#             "chart_product": chart_product,
#         })

#         # Prepare PDF response
#         response = HttpResponse(content_type='application/pdf')
#         response['Content-Disposition'] = f'attachment; filename="expense_report.pdf"'

#         pdf_file = BytesIO()
#         pisa_status = pisa.CreatePDF(src=html, dest=pdf_file, encoding='UTF-8')

#         if pisa_status.err:
#             return HttpResponse("Error generating PDF", status=500)

#         # Send PDF
#         response.write(pdf_file.getvalue())
#         return response

#     return HttpResponse("Invalid request", status=400)
import base64
from django.template.loader import render_to_string
from django.http import HttpResponse
from xhtml2pdf import pisa
from io import BytesIO
from .models import Expense


def generate_report(request):
    if request.method == 'POST':
        report_type = request.POST.get('report_type')
        year = request.POST.get('year')
        month = request.POST.get('month')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        category_chart = request.POST.get('category_chart_image')  # base64
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

        # Product data for the chart (Top 5 products)
        top_products = Expense.objects.values('product_name').annotate(total_spent=Sum('amount')).order_by('-total_spent')[:5]
        product_labels = [product['product_name'] for product in top_products]
        product_data = [product['total_spent'] for product in top_products]

        # Render PDF HTML
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

        # Generate PDF
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{title.replace(" ", "_")}.pdf"'

        pisa.CreatePDF(BytesIO(html.encode("UTF-8")), dest=response, encoding='UTF-8')
        return response
