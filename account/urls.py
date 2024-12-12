from django.urls import path
from .views import SignInView, SignUpView, UserPreferenceView

urlpatterns = [
    path('signup/', SignUpView.as_view(), name='sign-up'),
    path('signin/', SignInView.as_view(), name='sign-in'),
    path('preferences/', UserPreferenceView.as_view(), name='user-preferences')
]