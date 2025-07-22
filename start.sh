echo "🚀 Starting Ingredient Analysis API..."

# Check if running in production
if [ "$ENVIRONMENT" = "production" ]; then
    echo "📦 Running in production mode with Gunicorn"
    gunicorn app:app -c gunicorn.conf.py
else
    echo "🔧 Running in development mode with Uvicorn"
    uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --reload
fi
