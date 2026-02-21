"""
Django settings for traffic_analyzer project.
"""
import os
from pathlib import Path
import environ

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize django-environ
env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ['traffic-analysis-backend-9vjd.onrender.com', 'localhost', '127.0.0.1']),
    CORS_ALLOWED_ORIGINS=(list, ['http://localhost:5173', 'https://traffic-analysis-frontend-sandy.vercel.app']),
    MAX_FILE_SIZE=(int, 209715200),
    YOLO_MODEL_PATH=(str, str(BASE_DIR.parent / 'yolov8n.pt')),
    SECRET_KEY=(str, 'django-insecure-render-default-key-123456789'),
)

# Read .env file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['traffic-analysis-backend-9vjd.onrender.com', 'localhost', '127.0.0.1']

if isinstance(env('ALLOWED_HOSTS'), list):
    ALLOWED_HOSTS.extend(env('ALLOWED_HOSTS'))

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third party
    'rest_framework',
    'corsheaders',
    # Local apps
    'analysis',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'traffic_analyzer.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'traffic_analyzer.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (uploads, results)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

WHITENOISE_ROOT = MEDIA_ROOT

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS configuration
CORS_ALLOWED_ORIGINS = [
    'http://localhost:5173',
    'https://traffic-analysis-frontend-sandy.vercel.app'
]

if isinstance(env('CORS_ALLOWED_ORIGINS'), list):
    CORS_ALLOWED_ORIGINS.extend(env('CORS_ALLOWED_ORIGINS'))

CORS_ALLOW_CREDENTIALS = True

# Django REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 12,
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '200/day',
        'upload': '10/minute',
    },
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# File upload settings
MAX_FILE_SIZE = env('MAX_FILE_SIZE')
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_FILE_SIZE
FILE_UPLOAD_MAX_MEMORY_SIZE = MAX_FILE_SIZE

ALLOWED_UPLOAD_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}

ALLOWED_MIME_TYPES = {
    'image/jpeg': ['jpg', 'jpeg'],
    'image/png': ['png'],
    'video/mp4': ['mp4'],
    'video/x-msvideo': ['avi'],
    'video/quicktime': ['mov'],
}

# YOLO model path
YOLO_MODEL_PATH = env('YOLO_MODEL_PATH')

# Signal configuration defaults
DEFAULT_SIGNAL_CONFIG = {
    'green_min': 8,
    'green_max': 30,
    'yellow': 4,
    'red_min': 1,
    'red_max': 5,
    'min_green': 4,
    'hysteresis': 6,
}

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs' / 'traffic_analyzer.log',
            'maxBytes': 1_000_000,
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Create required directories
for d in [MEDIA_ROOT / 'uploads', MEDIA_ROOT / 'results', BASE_DIR / 'logs']:
    d.mkdir(parents=True, exist_ok=True)
