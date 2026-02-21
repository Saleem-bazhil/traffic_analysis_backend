#!/usr/bin/env bash

# Run database migrations before starting the web server
python manage.py migrate --no-input

# Start the gunicorn web server
gunicorn traffic_analyzer.wsgi
