from django import forms

class InvoiceUploadForm(forms.Form):
    image = forms.ImageField()

# expenses/forms.py
# from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']



from .models import Expense

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['date', 'category', 'amount', 'product_name']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }

