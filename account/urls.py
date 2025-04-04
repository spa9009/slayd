from django.urls import path
from .views import SignInView, SignUpView, UserPreferenceView
from instagram_bot.views import MetaWebhookView

urlpatterns = [
    path('signup/', SignUpView.as_view(), name='sign-up'),
    path('signin/', SignInView.as_view(), name='sign-in'),
    path('preferences/', UserPreferenceView.as_view(), name='user-preferences'),
    path('webhook/', MetaWebhookView.as_view(), name='meta_webhook_slash'),
    path('webhook', MetaWebhookView.as_view(), name='meta_webhook'),
]