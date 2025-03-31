from django.urls import path
from . import views
from .views import get_user_categories
from .views import get_user_years
from .views import generate_report

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.index, name='index'),
    path('confirm/', views.confirm, name='confirm'),
    path('recent-expenses/', views.recent_expenses, name='recent_expenses'),
    path('expense/edit/<int:expense_id>/', views.edit_expense, name='edit_expense'),
    path('expense/delete/<int:expense_id>/', views.delete_expense, name='delete_expense'),
    path('api/get-user-categories/', get_user_categories, name='get_user_categories'),
    path('api/get-user-years/', get_user_years, name='get_user_years'),
    path("generate-report/", generate_report, name="generate_report"),
]






