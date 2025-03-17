DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

ALLOWED_HOSTS = [
    'api.slayd.in',
    'localhost',
    '127.0.0.1',
]

# Add CORS settings if needed
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOWED_ORIGINS = [
    "https://api.slayd.in",
]

# Add your domain to CSRF trusted origins
CSRF_TRUSTED_ORIGINS = [
    "https://api.slayd.in",
]

# Make sure your Instagram token is set
INSTAGRAM_ACCESS_TOKEN = "IGAAhRitqpc7JBZAE1Jb0JUR3FsZAjhpdHhTTER3dFhsQmJxNDhEVzNkZAEtBSFhfdEl5QldCQkVjaUFyTDdobDg1Y3BDTm5YUkloZAHpFQ0ZAleU03UlFhU3BIV1YyQW9vQ1FZASElHc3dOZAUxkQmNnRkNiTEs5M2FTTDNnV3BkY1JkVQZDZD"
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
} 