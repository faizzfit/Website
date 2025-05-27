from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user, user_logged_in
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask.sessions import SessionInterface, SessionMixin
from flask_migrate import Migrate
from uuid import uuid4
from datetime import datetime
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import cv2
import numpy as np
import hashlib
import exifread
from io import BytesIO
from fpdf import FPDF
import io
import pytz
import json

class CustomSession(dict, SessionMixin):
    def __init__(self):
        super().__init__()
        self._id = str(uuid4())

class CustomSessionInterface(SessionInterface):
    def open_session(self, app, request):
        session = CustomSession()
        return session

    def save_session(self, app, session, response):
        cookie_options = {
            'httponly': True,
            'secure': False,  # Set to True if using HTTPS
            'samesite': 'Lax'
        }
        if session:
            response.set_cookie(
                'session',  # Using 'session' as the cookie name
                session.get('_id', ''),
                **cookie_options
            )

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Use a fixed secret key for development
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SESSION_COOKIE_NAME'] = 'session'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Ensure upload directory exists and is writable
upload_dir = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_dir

try:
    print(f"Checking upload directory: {upload_dir}")
    os.makedirs(upload_dir, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(upload_dir, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print("Upload directory is ready and writable")
except Exception as e:
    print(f"Error with upload directory: {str(e)}")
    upload_dir = None

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.session_interface = CustomSessionInterface()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # Add is_admin field

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Add this after the User model definition
class UserLogin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Kuala_Lumpur')))
    ip_address = db.Column(db.String(50))
    user_agent = db.Column(db.String(200))
    
    # Add relationship to User model
    user = db.relationship('User', backref=db.backref('logins', lazy=True))

# Add UserLogout model
class UserLogout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Kuala_Lumpur')))
    ip_address = db.Column(db.String(50))
    user_agent = db.Column(db.String(200))
    
    # Add relationship to User model
    user = db.relationship('User', backref=db.backref('logouts', lazy=True))

# Model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        for idx, layer in enumerate(self.network):
            x = layer(x)
            if idx == 2:
                self.activation1 = x
            elif idx == 5:
                self.activation2 = x
            elif idx == 8:
                self.activation3 = x
        return x

# Load model
try:
    print("Loading model...")
    model = SimpleCNN().to(device)
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Image processing functions
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def apply_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    temp_path = str(image_path).replace('.jpg', '_temp.jpg')
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path)
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_path)
    return ela_image

def save_activation_grid(activation, title, out_path):
    act = activation.squeeze(0).cpu().detach().numpy()
    fig, axes = plt.subplots(1, min(16, act.shape[0]), figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(act[i], cmap='viridis')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def generate_gradcam(input_tensor, model, target_layer, output_path, original_image):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output.backward()

    grads = gradients[0].squeeze(0).cpu().detach().numpy()
    acts = activations[0].squeeze(0).cpu().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts[0].shape)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (original_image.width, original_image.height))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_image), 0.6, heatmap, 0.4, 0)

    plt.figure()
    plt.imshow(overlay)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

    handle_fw.remove()
    handle_bw.remove()

def analyze_image(image_path):
    try:
        if model is None:
            raise RuntimeError("Model not properly initialized. Please check model loading.")
        
        # Verify original file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        print(f"Loading image into memory: {image_path}")
        # Load the image into memory
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        # Create BytesIO object for in-memory processing
        image_buffer = BytesIO(image_data)
        pil_image = Image.open(image_buffer).convert('RGB')
        
        print("Starting ELA analysis")
        # Apply ELA in memory
        temp_buffer = BytesIO()
        pil_image.save(temp_buffer, format='JPEG', quality=90)
        temp_buffer.seek(0)
        compressed = Image.open(temp_buffer)
        ela_image = ImageChops.difference(pil_image, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Save ELA image in main uploads directory
        ela_filename = os.path.splitext(os.path.basename(image_path))[0] + '_ela.png'
        ela_path = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
        ela_image.save(ela_path)
        print("ELA analysis complete")
        
        print("Starting model prediction")
        # Prepare for prediction
        ela_tensor = transform(ela_image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(ela_tensor)
            raw_score = output.item()
            prediction = 'Fake' if raw_score > 0.5 else 'Real'
            confidence = raw_score * 100 if prediction == 'Fake' else (1 - raw_score) * 100
        print(f"Prediction complete: {prediction} ({confidence:.2f}%)")
        
        print("Generating activation visualizations")
        # Save activation visualizations in main uploads directory
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        activation_paths = []
        for i, activation in enumerate([model.activation1, model.activation2, model.activation3], 1):
            act_filename = f"{base_name}_activation{i}.png"
            act_path = os.path.join(app.config['UPLOAD_FOLDER'], act_filename)
            save_activation_grid(activation, f"Layer {i} Activations", act_path)
            activation_paths.append(act_filename)
        print("Activation visualizations complete")
        
        print("Generating Grad-CAM")
        # Generate and save Grad-CAM in main uploads directory
        gradcam_filename = f"{base_name}_gradcam.png"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        generate_gradcam(ela_tensor, model, model.network[6], gradcam_path, pil_image)
        print("Grad-CAM complete")
        
        print("Extracting metadata")
        # Extract metadata
        def get_hash(data, algo='sha256'):
            h = hashlib.new(algo)
            h.update(data)
            return h.hexdigest()
        
        # Get basic file info
        stat = os.stat(image_path)
        
        # Get EXIF data
        exif_data = {}
        try:
            image_buffer.seek(0)
            tags = exifread.process_file(image_buffer, details=False)
            if tags:
                exif_data = {str(tag): str(tags[tag]) for tag in tags.keys()}
            else:
                exif_data = "No EXIF metadata found in this image."
        except Exception as e:
            exif_data = f"Error reading EXIF metadata: {str(e)}"
        
        metadata = {
            'format': pil_image.format,
            'mode': pil_image.mode,
            'dimensions': f"{pil_image.width} x {pil_image.height} pixels",
            'file_size': f"{stat.st_size:,} bytes",
            'created_time': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'sha256': get_hash(image_data, 'sha256'),
            'md5': get_hash(image_data, 'md5'),
            'exif': exif_data
        }
        print("Metadata extraction complete")
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'original_image': os.path.basename(image_path),
            'ela_image': ela_filename,
            'activation1': activation_paths[0],
            'activation2': activation_paths[1],
            'activation3': activation_paths[2],
            'gradcam': gradcam_filename,
            'metadata': metadata,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        print("Analysis complete, returning results")
        return result
        
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_image = db.Column(db.String(255), nullable=False)
    ela_image = db.Column(db.String(255), nullable=False)
    activation1 = db.Column(db.String(255), nullable=False)
    activation2 = db.Column(db.String(255), nullable=False)
    activation3 = db.Column(db.String(255), nullable=False)
    gradcam = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Kuala_Lumpur')))
    
    # Metadata fields
    image_format = db.Column(db.String(50))
    image_mode = db.Column(db.String(50))
    dimensions = db.Column(db.String(100))
    file_size = db.Column(db.String(100))
    created_time = db.Column(db.String(100))
    modified_time = db.Column(db.String(100))
    sha256_hash = db.Column(db.String(64))
    md5_hash = db.Column(db.String(32))
    exif_data = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt for username: {username}")  # Debug log
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True)
            print(f"Login successful for user: {username}")  # Debug log
            print(f"Is authenticated: {current_user.is_authenticated}")  # Debug log
            
            # Redirect logic
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            print(f"Login failed for username: {username}")  # Debug log
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Enforce password requirements
        if len(password) < 8 or not any(c.isdigit() for c in password) or not any(c.isalpha() for c in password):
            flash('Password must be at least 8 characters long and contain both letters and numbers.')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        print("POST request received")
        
        if upload_dir is None:
            flash('Upload directory not available. Please contact administrator.', 'error')
            return redirect(url_for('dashboard'))
            
        if 'file' not in request.files:
            print("No file in request")
            flash('No file selected', 'error')
            return redirect(url_for('dashboard'))
        
        file = request.files['file']
        if file.filename == '':
            print("Empty filename")
            flash('No file selected', 'error')
            return redirect(url_for('dashboard'))
        
        if file and allowed_file(file.filename):
            try:
                print(f"Processing file: {file.filename}")
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Create upload directory if it doesn't exist
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save the uploaded file
                file.save(filepath)
                print(f"File saved to: {filepath}")
                
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Failed to save uploaded file: {filepath}")
                
                # Analyze the image
                print("Starting image analysis...")
                result = analyze_image(filepath)
                print(f"Analysis complete. Result structure: {list(result.keys())}")
                
                # Save the result to database with metadata
                print("Saving result to database...")
                scan_result = ScanResult(
                    user_id=current_user.id,
                    original_image=result['original_image'],
                    ela_image=result['ela_image'],
                    activation1=result['activation1'],
                    activation2=result['activation2'],
                    activation3=result['activation3'],
                    gradcam=result['gradcam'],
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    image_format=result['metadata']['format'],
                    image_mode=result['metadata']['mode'],
                    dimensions=result['metadata']['dimensions'],
                    file_size=result['metadata']['file_size'],
                    created_time=result['metadata']['created_time'],
                    modified_time=result['metadata']['modified_time'],
                    sha256_hash=result['metadata']['sha256'],
                    md5_hash=result['metadata']['md5'],
                    exif_data=str(result['metadata']['exif'])
                )
                db.session.add(scan_result)
                db.session.commit()
                print("Result saved to database successfully")
                
                # Structure the metadata for the template
                print("Preparing result for template...")
                result['metadata'] = {
                    'format': scan_result.image_format,
                    'mode': scan_result.image_mode,
                    'dimensions': scan_result.dimensions,
                    'file_size': scan_result.file_size,
                    'created_time': scan_result.created_time,
                    'modified_time': scan_result.modified_time,
                    'sha256': scan_result.sha256_hash,
                    'md5': scan_result.md5_hash,
                    'exif': json.loads(scan_result.exif_data) if scan_result.exif_data.strip().startswith('{') else scan_result.exif_data
                }
                # Add the scan_result id to result for PDF download
                result['id'] = scan_result.id
                print("Result prepared for template")
                
                # Return the template with results
                print("Rendering template with results...")
                return render_template('dashboard.html', result=result)
                
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
                print("Full error details:")
                import traceback
                traceback.print_exc()
                
                # Clean up any partially processed files
                try:
                    if 'filepath' in locals() and os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"Cleaned up uploaded file: {filepath}")
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {str(cleanup_error)}")
                
                flash(f'Error analyzing image: {str(e)}', 'error')
                return redirect(url_for('dashboard'))
        else:
            print(f"Invalid file type: {file.filename}")
            flash('Invalid file type. Please upload a JPG or PNG image.', 'error')
            return redirect(url_for('dashboard'))
    
    print("GET request - rendering empty dashboard")
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    # Record logout activity
    try:
        logout_record = UserLogout(
            user_id=current_user.id,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string,
            timestamp=datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
        )
        db.session.add(logout_record)
        db.session.commit()
    except Exception as e:
        print(f"Error creating logout record: {str(e)}")
        db.session.rollback()
    
    logout_user()
    return redirect(url_for('index'))

@app.route('/report')
@login_required
def report():
    scan_results = ScanResult.query.filter_by(user_id=current_user.id).order_by(ScanResult.timestamp.desc()).all()
    return render_template('report.html', reports=scan_results)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))
    
    # Get all users
    users = User.query.all()
    
    # Get total scans
    total_scans = ScanResult.query.count()
    
    # Get fake image count
    fake_count = ScanResult.query.filter_by(prediction='Fake').count()
    
    # Get recent logins and logouts with user information
    recent_logins = db.session.query(UserLogin, User).join(User).order_by(UserLogin.timestamp.desc()).limit(50).all()
    recent_logouts = db.session.query(UserLogout, User).join(User).order_by(UserLogout.timestamp.desc()).limit(50).all()
    
    # Combine and sort login/logout activities
    activities = []
    for login, user in recent_logins:
        activities.append({
            'type': 'login',
            'user': user,
            'timestamp': login.timestamp,
            'ip_address': login.ip_address,
            'user_agent': login.user_agent
        })
    for logout, user in recent_logouts:
        activities.append({
            'type': 'logout',
            'user': user,
            'timestamp': logout.timestamp,
            'ip_address': logout.ip_address,
            'user_agent': logout.user_agent
        })
    
    # Sort activities by timestamp
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    activities = activities[:50]  # Keep only the 50 most recent activities
    
    # Get all scans with user information
    all_scans = db.session.query(ScanResult, User).join(User).order_by(ScanResult.timestamp.desc()).all()
    
    return render_template('admin_dashboard.html', 
                         users=users,
                         total_scans=total_scans,
                         fake_count=fake_count,
                         activities=activities,
                         all_scans=all_scans)

@app.route('/add_user', methods=['POST'])
@login_required
def add_user():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    is_admin = data.get('is_admin', False)
    
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists'})
    
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered'})
    
    user = User(username=username, email=email, is_admin=is_admin)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/edit_user', methods=['POST'])
@login_required
def edit_user():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    data = request.get_json()
    user_id = data.get('user_id')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.get_or_404(user_id)
    
    # Check if username is taken by another user
    existing_user = User.query.filter_by(username=username).first()
    if existing_user and existing_user.id != user.id:
        return jsonify({'success': False, 'message': 'Username already exists'})
    
    # Check if email is taken by another user
    existing_user = User.query.filter_by(email=email).first()
    if existing_user and existing_user.id != user.id:
        return jsonify({'success': False, 'message': 'Email already registered'})
    
    user.username = username
    user.email = email
    if password:  # Only update password if a new one is provided
        # Enforce password requirements
        if len(password) < 8 or not any(c.isdigit() for c in password) or not any(c.isalpha() for c in password):
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long and contain both letters and numbers.'})
        user.set_password(password)
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    scan = ScanResult.query.get_or_404(scan_id)
    
    # Delete associated files
    try:
        files_to_delete = [
            scan.original_image,
            scan.ela_image,
            scan.activation1,
            scan.activation2,
            scan.activation3,
            scan.gradcam
        ]
        
        for file in files_to_delete:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error deleting files: {str(e)}")
    
    # Delete from database
    db.session.delete(scan)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/generate_pdf_report/<int:report_id>')
@login_required
def generate_pdf_report(report_id):
    report = ScanResult.query.get_or_404(report_id)
    if report.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Create PDF with A4 size
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)  # Set margins to 10mm on all sides
    pdf.set_auto_page_break(False)  # Disable automatic page breaks
    
    # Add first page
    pdf.add_page()
    
    # Add logo/header
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(33, 33, 33)  # Dark gray
    pdf.cell(0, 20, "SecureAuth", ln=True, align='C')
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI-Generated Image Analysis Report", ln=True, align='C')
    
    # Add report details with better formatting
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)  # Gray
    pdf.cell(0, 5, f"Report ID: {report.id}", ln=True, align='C')
    pdf.cell(0, 5, f"Generated on: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.cell(0, 5, f"Analysis performed by: {current_user.username}", ln=True, align='C')
    pdf.ln(10)
    
    # Add a line separator
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Analysis Result with better styling
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 10, "Analysis Result", ln=True)
    
    # Create a colored box for the result
    result_color = (194, 40, 40) if report.prediction == 'Fake' else (46, 125, 50)
    pdf.set_fill_color(*result_color)
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, f"{report.prediction}", ln=True, align='C', fill=True)
    
    # Confidence score
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, f"Confidence: {report.confidence:.2f}%", ln=True, align='C')
    pdf.ln(10)
    
    # Add a helper function to insert a description
    def add_description(text):
        pdf.set_text_color(100, 100, 100)  # Gray
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 5, text)
        pdf.ln(5)

    # Add images with fixed sizing and layout
    def add_image_section(image_path, title, description, box_size=95.4):  # Decreased from 105.4mm to 95.4mm (105.4 - 10mm)
        from PIL import Image  # Ensure Image is available in the local scope
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], image_path)):
            # Check if there's enough space left on the page
            y_now = pdf.get_y()
            page_width = 210  # A4 width in mm
            page_height = 297  # A4 height in mm
            bottom_margin = 15
            needed_space = 10 + 2 + box_size + 2 + 10 + 5  # title + gap + box + gap + desc + gap
            if y_now + needed_space > page_height - bottom_margin:
                pdf.add_page()
                y_now = pdf.get_y()
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(33, 33, 33)
            pdf.cell(0, 10, title, ln=True, align='C')
            pdf.ln(2)
            # Draw border box centered
            x_box = (page_width - box_size) / 2
            y_now = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.rect(x_box, y_now, box_size, box_size)
            # Add image centered in box
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
            img = Image.open(img_path)
            img_w, img_h = img.size
            aspect = img_w / img_h
            if aspect > 1:
                w = box_size
                h = box_size / aspect
            else:
                h = box_size
                w = box_size * aspect
            x_img = x_box + (box_size - w) / 2
            y_img = y_now + (box_size - h) / 2
            pdf.image(img_path, x=x_img, y=y_img, w=w, h=h)
            pdf.ln(box_size + 2)
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, description, align='C')
            pdf.ln(5)

    # Original Image Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Image Analysis", ln=True)
    pdf.ln(5)
    
    # Add original image
    add_image_section(report.original_image, "Original Image", "This is the original image uploaded for analysis. It is used as a reference for the analysis results.")
    pdf.ln(5)
    
    # Add ELA image
    add_image_section(report.ela_image, "Error Level Analysis (ELA)", "Error Level Analysis (ELA) highlights areas of the image that have been modified. Higher error levels (brighter areas) may indicate tampering or editing.")
    pdf.ln(5)

    # Add Grad-CAM heatmap with same size as ELA
    add_image_section(report.gradcam, "AI Attention Visualization (Grad-CAM)", "Grad-CAM heatmap shows which regions of the image the AI model focused on to make its prediction. Brighter areas indicate regions of higher importance.")
    
    # Add new page for neural network analysis
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Neural Network Analysis", ln=True, align='C')
    pdf.ln(4)

    # Process activation layers on one page
    activation_layers = [
        ("Layer 1 Activations", report.activation1, "Layer 1 activations show the initial features detected by the neural network, such as edges and textures."),
        ("Layer 2 Activations", report.activation2, "Layer 2 activations reveal more complex patterns and structures identified by the network."),
        ("Layer 3 Activations", report.activation3, "Layer 3 activations represent high-level features that help the network make its final prediction.")
    ]

    # Calculate available height for each visualization
    page_height = 250  # Leave some margin at bottom
    header_height = 20  # Space for title and description
    total_headers = len(activation_layers) * header_height
    available_height = (page_height - total_headers) / len(activation_layers)
    page_width = 190  # Leave margins

    for title, img, desc in activation_layers:
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], img)):
            # Add title
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(33, 33, 33)
            pdf.cell(0, 8, title, ln=True, align='C')
            pdf.ln(2)

            # Load and calculate image dimensions
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img)
            im = Image.open(img_path)
            img_w, img_h = im.size
            
            # Calculate scaling to fit both width and height constraints
            width_scale = page_width / img_w
            height_scale = available_height / img_h
            scale = min(width_scale, height_scale)  # Use the smaller scale to fit both dimensions
            
            new_width = img_w * scale
            new_height = img_h * scale

            # Add image centered on page
            x_img = (210 - new_width) / 2  # Center horizontally
            pdf.image(img_path, x=x_img, y=pdf.get_y(), w=new_width, h=new_height)
            
            # Add description below image
            pdf.ln(new_height + 2)
            pdf.set_font("Arial", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 4, desc, align='C')
            pdf.ln(5)

    # Add new page for metadata
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Technical Metadata", ln=True)
    pdf.ln(5)
    
    # Create a table-like layout for metadata
    def add_metadata_row(label, value, is_bold=False):
        pdf.set_font("Arial", "B" if is_bold else "", 10)
        pdf.cell(60, 8, label, border=1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, str(value), border=1, ln=True)
    
    # Basic Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Basic Information", ln=True)
    pdf.ln(2)
    
    metadata = {
        'Format': report.image_format,
        'Mode': report.image_mode,
        'Dimensions': report.dimensions,
        'File Size': report.file_size,
        'Created': report.created_time,
        'Modified': report.modified_time
    }
    
    for key, value in metadata.items():
        add_metadata_row(key + ":", value)
    
    pdf.ln(5)
    
    # File Hashes
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "File Hashes", ln=True)
    pdf.ln(2)
    
    add_metadata_row("SHA-256:", report.sha256_hash)
    add_metadata_row("MD5:", report.md5_hash)
    
    # EXIF Data
    if report.exif_data and report.exif_data != "No EXIF metadata found in this image.":
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "EXIF Data", ln=True)
        pdf.ln(2)
        
        try:
            exif_data = json.loads(report.exif_data) if report.exif_data.strip().startswith('{') else report.exif_data
            if isinstance(exif_data, dict):
                for key, value in exif_data.items():
                    add_metadata_row(str(key) + ":", str(value))
            else:
                add_metadata_row("EXIF Data:", exif_data)
        except Exception:
            add_metadata_row("EXIF Data:", report.exif_data)
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "This report was generated by an AI-powered image analysis system.", ln=True, align='C')
    pdf.cell(0, 5, f"© {datetime.now().year} SecureAuth. All rights reserved.", ln=True, align='C')
    
    # Generate PDF
    pdf_output = pdf.output(dest='S').encode('latin1')
    
    # Create response
    response = make_response(pdf_output)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=report_{report.id}.pdf'
    
    return response

# Configure matplotlib for server environment
plt.ioff()  # Turn off interactive mode
print("Matplotlib configured for server environment")

# Add signal handler for login tracking
@user_logged_in.connect_via(app)
def track_login(sender, user, **extra):
    print(f"Login signal received for user: {user.username}")  # Debug log
    try:
        login_record = UserLogin(
            user_id=user.id,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string,
            timestamp=datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
        )
        db.session.add(login_record)
        db.session.commit()
        print(f"Login record created successfully for user: {user.username}")  # Debug log
    except Exception as e:
        print(f"Error creating login record: {str(e)}")  # Debug log
        db.session.rollback()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create a test admin user if it doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(username='admin', email='admin@example.com', is_admin=True)
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            print("Created admin user: admin/admin123")
    
    app.run(debug=True) 
