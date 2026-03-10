# Django Machine Learning Project

A Django web application that integrates machine learning models for data analysis and prediction.

## Features

- **Classification Models**: Machine learning classification algorithms
- **Clustering Models**: Data clustering and analysis
- **Regression Models**: Predictive regression models
- **Data Visualization**: Interactive charts and visualizations
- **Django Integration**: Web-based interface for ML operations

## Project Structure

```
django_ml_project/
├── config/                 # Django project configuration
├── predictor/              # Main Django app
│   ├── models.py          # Django models
│   ├── views.py           # View functions
│   ├── urls.py            # URL routing
│   └── ...
├── model_generators/      # ML model training scripts
│   ├── classification/    # Classification models
│   ├── clustering/        # Clustering algorithms
│   └── regression/        # Regression models
├── dummy-data/           # Sample datasets
├── requirements.txt      # Python dependencies
└── manage.py            # Django management script
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run database migrations:
```bash
python manage.py migrate
```

4. Create superuser (optional):
```bash
python manage.py createsuperuser
```

## Usage

### Development Server
```bash
python manage.py runserver
```

### Training Models
Navigate to the respective model directories to train different ML models:

- **Classification**: `python model_generators/classification/train_classifier.py`
- **Clustering**: `python model_generators/clustering/train_cluster.py`
- **Regression**: `python model_generators/regression/train_regression.py`

## Dependencies

- Django 6.0.3
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- plotly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes as part of a Machine Learning course.

## Author
Made with 💓 by Joyeuse Iradukunda
