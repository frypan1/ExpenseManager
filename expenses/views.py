# from django.shortcuts import render, redirect
# from django.contrib.auth import login, authenticate
# from django.contrib.auth.forms import AuthenticationForm
# from django.contrib.auth.decorators import login_required
# from django.contrib import messages
# from django.contrib.auth import logout
# from .forms import RegisterForm, InvoiceUploadForm
# from .models import Expense, Category
# from .utils import perform_ocr
# from datetime import datetime, timedelta
# from django.db.models import Sum
# import openai
# from django.conf import settings
# from django.utils.timezone import now


# openai.api_key = settings.OPENAI_API_KEY

# # Registration view
# def register(request):
#     if request.method == 'POST':
#         form = RegisterForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user)  # Log the user in after successful registration
#             messages.success(request, "Account created successfully!")
#             return redirect('index')  # Redirect to the index page after registration
#     else:
#         form = RegisterForm()
#     return render(request, 'registration/register.html', {'form': form})

# # Login view
# def login_view(request):
#     if request.method == 'POST':
#         form = AuthenticationForm(request, data=request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             login(request, user)
#             return redirect('index')  # Redirect to the index page after login
#         else:
#             messages.error(request, "Invalid credentials")
#     else:
#         form = AuthenticationForm()
#     return render(request, 'registration/login.html', {'form': form})

# # Logout view
# def logout_view(request):
#     logout(request)
#     return redirect('login')  # Redirect to login page after logging out

# # Dashboard view (user-specific data)
# # views.py (Make sure you are filtering by the user in the view)

# @login_required
# def index(request):
#     # Fetch the total amount spent per category from the Expense model, filtering by the logged-in user
#     expenses_by_category = Expense.objects.filter(user=request.user).values('category').annotate(total_amount=Sum('amount')).order_by('category')
    
#     # Prepare the labels and data for the chart (category names and total amounts)
#     labels = [expense['category'] for expense in expenses_by_category]
#     data = [float(expense['total_amount']) for expense in expenses_by_category]

#     if request.method == 'POST':
#         form = InvoiceUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             file = form.cleaned_data['image']
#             file_path = 'media/uploads/' + file.name
#             with open(file_path, 'wb') as f:
#                 for chunk in file.chunks():
#                     f.write(chunk)
            
#             # Perform OCR on the uploaded image to extract text
#             extracted_text = perform_ocr(file_path)
#             print("Extracted word count: ", len(extracted_text))
#             extracted_details_text = extract_invoice_details(extracted_text)
#             print(extracted_details_text)
            
#             if extracted_details_text:
#                 extracted_entries = []
#                 for line in extracted_details_text.splitlines():
#                     if line.strip():
#                         entry = {}
#                         for item in line.split(", "):
#                             key, value = item.split(": ")
#                             entry[key.strip().lower().replace(" ", "_")] = value.strip()
#                         extracted_entries.append(entry)

#                 # Fetch all categories from the database
#                 fixed_categories = list(Category.objects.values_list('name', flat=True))

#                 return render(request, 'review.html', {
#                     'extracted_entries': extracted_entries,
#                     'fixed_categories': fixed_categories
#                 })

#     else:
#         form = InvoiceUploadForm()

#     # Pass the form, labels, and data for the pie chart to the template
#     context = {
#         'form': form,
#         'labels': labels,
#         'data': data,
#     }

#     return render(request, 'index.html', context)


# # Extract invoice details using OpenAI API
# def extract_invoice_details(invoice_text):
#     try:
#         categories = Category.objects.values_list('name', flat=True)
#         category_list = ", ".join([f"'{category}'" for category in categories])
        
#         prompt = f"""
#         Extract the following details from the invoice:
#         1. Date of purchase
#         2. Suggested category from this list: [{category_list}]
#         3. Amount spent
#         4. Product or service name

#         Invoice text:
#         {invoice_text}

#         Return the details in the following format (one entry per line):
#         Date: DD-MM-YYYY, Category: <suggested_category>, Amount: <numeric_amount>, Product Name: <product_name>.
#         Only return the numeric value for the amount without currency symbols.
#         Remove commas from product name.
#         """
        
#         response = openai.Completion.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             max_tokens=300,
#             temperature=0.3
#         )
        
#         return response.choices[0].text.strip()
    
#     except Exception as e:
#         print(f"Error extracting invoice details: {e}")
#         return None

# # Confirmation view (saving the extracted details into the database)
# def confirm(request):
#     if request.method == 'POST':
#         entry_count = int(request.POST.get('entry_count', 0))
#         fixed_categories = ["Groceries", "Entertainment", "Dining", "Transportation"]
#         new_categories = []
#         expense_details_list = []

#         for i in range(1, entry_count + 1):
#             date_str = request.POST.get(f'date_{i}')
#             selected_category = request.POST.get(f'category_{i}')
#             custom_category = request.POST.get(f'custom_category_{i}')
#             amount = request.POST.get(f'amount_{i}')
#             product_name = request.POST.get(f'product_name_{i}')

#             category = custom_category if selected_category == "Custom" and custom_category else selected_category

#             if category not in fixed_categories and category:
#                 new_categories.append(category)
#                 fixed_categories.append(category)

#             date_obj = datetime.strptime(date_str, '%d-%m-%Y').date()

#             expense_details_list.append({
#                 'date': date_obj,
#                 'category': category,
#                 'amount': float(amount),
#                 'product_name': product_name
#             })

#         for details in expense_details_list:
#             invoice = Expense(
#                 user=request.user,  # Save the expense for the logged-in user
#                 date=details['date'],
#                 category=details['category'],
#                 amount=details['amount'],
#                 product_name=details['product_name']
#             )
#             invoice.save()

#         for category in new_categories:
#             Category.objects.get_or_create(name=category)
        
#         return redirect('index')
    

# @login_required
# def recent_expenses(request):
#     # Get the current date and calculate the date 30 days ago
#     today = now().date()
#     thirty_days_ago = today - timedelta(days=30)

#     # Filter expenses for the logged-in user within the last 30 days
#     recent_expenses = Expense.objects.filter(
#         user=request.user,
#         date__range=[thirty_days_ago, today]
#     ).order_by('-date')  # Order by most recent first

#     # Pass the filtered expenses to the template
#     return render(request, 'recent_expenses.html', {'recent_expenses': recent_expenses})

# from django.shortcuts import render, get_object_or_404, redirect
# from .models import Expense
# from .forms import ExpenseForm
# from django.contrib.auth.decorators import login_required

# @login_required
# def edit_expense(request, expense_id):
#     expense = get_object_or_404(Expense, id=expense_id, user=request.user)
#     if request.method == 'POST':
#         form = ExpenseForm(request.POST, instance=expense)
#         if form.is_valid():
#             form.save()
#             return redirect('recent_expenses')  # Redirect to the recent expenses page
#     else:
#         form = ExpenseForm(instance=expense)
#     return render(request, 'edit_expense.html', {'form': form})


# @login_required
# def delete_expense(request, expense_id):
#     expense = get_object_or_404(Expense, id=expense_id, user=request.user)
#     if request.method == 'POST':
#         expense.delete()
#         return redirect('recent_expenses')
#     return render(request, 'confirm_delete.html', {'expense': expense})

'''from django.shortcuts import render, redirect, get_object_or_404
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
@login_required
def index(request):
    # Fetch the total amount spent per category, filtering by the logged-in user
    expenses_by_category = (
        Expense.objects.filter(user=request.user)
        .values('category__name')
        .annotate(total_amount=Sum('amount'))
        .order_by('category__name')
    )

    # Prepare the labels and data for the chart (category names and total amounts)
    labels = [expense['category__name'] for expense in expenses_by_category]
    data = [float(expense['total_amount']) for expense in expenses_by_category]

    if request.method == 'POST':
        form = InvoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['image']
            file_path = f'media/uploads/{file.name}'
            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # Perform OCR on the uploaded image to extract text
            extracted_text = perform_ocr(file_path)
            extracted_details_text = extract_invoice_details(extracted_text)

            if extracted_details_text:
                extracted_entries = []
                for line in extracted_details_text.splitlines():
                    if line.strip():
                        entry = {}
                        for item in line.split(", "):
                            key, value = item.split(": ")
                            entry[key.strip().lower().replace(" ", "_")] = value.strip()
                        extracted_entries.append(entry)

                # Fetch user-specific categories and global categories
                user_categories = list(Category.objects.filter(user=request.user).values_list('name', flat=True))
                global_categories = list(Category.objects.filter(user__isnull=True).values_list('name', flat=True))
                fixed_categories = user_categories + global_categories

                return render(request, 'review.html', {
                    'extracted_entries': extracted_entries,
                    'fixed_categories': fixed_categories,
                })

    else:
        form = InvoiceUploadForm()

    context = {
        'form': form,
        'labels': labels,
        'data': data,
    }
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
        return response.choices[0].text.strip()

    except Exception as e:
        print(f"Error extracting invoice details: {e}")
        return None


# Confirmation view (saving the extracted details into the database)
@login_required
def confirm(request):
    if request.method == 'POST':
        entry_count = int(request.POST.get('entry_count', 0))
        new_categories = []
        expense_details_list = []

        for i in range(1, entry_count + 1):
            date_str = request.POST.get(f'date_{i}')
            selected_category = request.POST.get(f'category_{i}')
            custom_category = request.POST.get(f'custom_category_{i}')
            amount = request.POST.get(f'amount_{i}')
            product_name = request.POST.get(f'product_name_{i}')

            category_name = custom_category if selected_category == "Custom" and custom_category else selected_category

            # Create new category if it doesn't exist
            category, created = Category.objects.get_or_create(
                name=category_name,
                user=request.user,
            )
            if created:
                new_categories.append(category.name)

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
    return render(request, 'confirm_delete.html', {'expense': expense})'''


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
@login_required
def index(request):
    # Fetch the total amount spent per category, filtering by the logged-in user
    expenses_by_category = (
        Expense.objects.filter(user=request.user)
        .values('category__name')
        .annotate(total_amount=Sum('amount'))
        .order_by('category__name')
    )

    # Prepare the labels and data for the chart (category names and total amounts)
    labels = [expense['category__name'] for expense in expenses_by_category]
    data = [float(expense['total_amount']) for expense in expenses_by_category]

    if request.method == 'POST':
        form = InvoiceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['image']
            file_path = f'media/uploads/{file.name}'
            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # Perform OCR on the uploaded image to extract text
            extracted_text = perform_ocr(file_path)
            print(extracted_text)
            extracted_details_text = extract_invoice_details(extracted_text)

            if extracted_details_text:
                extracted_entries = []
                for line in extracted_details_text.splitlines():
                    if line.strip():
                        entry = {}
                        for item in line.split(", "):
                            key, value = item.split(": ")
                            entry[key.strip().lower().replace(" ", "_")] = value.strip()
                        extracted_entries.append(entry)

                # Fetch user-specific and global categories
                user_categories = list(Category.objects.filter(user=request.user).values_list('name', flat=True))
                global_categories = list(Category.objects.filter(user__isnull=True).values_list('name', flat=True))
                fixed_categories = user_categories + global_categories

                # Match extracted categories with existing categories
                for entry in extracted_entries:
                    if 'category' in entry and entry['category'] not in fixed_categories:
                        # If category is not in the list, assign 'Uncategorized' or flag it
                        entry['category'] = 'Uncategorized'

                return render(request, 'review.html', {
                    'extracted_entries': extracted_entries,
                    'fixed_categories': fixed_categories,
                })

    else:
        form = InvoiceUploadForm()

    context = {
        'form': form,
        'labels': labels,
        'data': data,
    }
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

#Hello World