# Image Authentication System

A Flask-based web application for detecting fake/real images using deep learning.

## Features

- User authentication and authorization
- Image upload and analysis
- Admin dashboard for user management
- Scan history tracking
- PDF report generation
- Activity logging

## Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following variables:
```
SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
```

5. Initialize the database:
```bash
flask db upgrade
```

6. Run the application:
```bash
flask run
```

## Project Structure

- `app.py`: Main application file
- `templates/`: HTML templates
- `static/`: Static files (CSS, JS, uploads)
- `migrations/`: Database migrations
- `model.pth`: Trained model file
- `predict_single.py`: Image prediction module

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details 