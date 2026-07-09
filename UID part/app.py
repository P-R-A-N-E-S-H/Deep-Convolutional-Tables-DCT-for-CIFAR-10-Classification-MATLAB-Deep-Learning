from flask import Flask, render_template, g
import sqlite3
import os
from PIL import Image
from torchvision.datasets import CIFAR10

app = Flask(__name__)
DATABASE = 'cifar_data.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def prepare_dataset():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cifar_images'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        # Create the table with the same schema as in setup_db.py
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cifar_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                filename TEXT NOT NULL,
                model_used TEXT,
                accuracy REAL
            )
        """)
        print("Created new table with schema from setup_db.py")

    # Check if we need to populate the dataset
    cursor.execute("SELECT COUNT(*) FROM cifar_images")
    count = cursor.fetchone()[0]

    if count < 10:  # We only have sample data or no data, need to load real dataset
        print("Loading CIFAR-10 dataset...")

        dataset = CIFAR10(root='cifar_data', train=True, download=True)
        label_names = dataset.classes  # ['airplane', 'automobile', ..., 'truck']

        os.makedirs('static/cifar10_images', exist_ok=True)

        for idx in range(len(dataset)):
            image, label_id = dataset[idx]
            image_name = f'image_{idx}.png'
            full_path = f'static/cifar10_images/{image_name}'

            image.save(full_path)
            
            # Insert using only the columns that exist in the table
            cursor.execute("""
                INSERT INTO cifar_images 
                (label, filename, model_used, accuracy) 
                VALUES (?, ?, ?, ?)
            """, (label_names[label_id], image_name, None, None))

            if idx % 1000 == 0:
                print(f"Saved {idx} images...")
                conn.commit()  # Commit every 1000 images to avoid large transactions

        conn.commit()
        print("Dataset preparation complete!")
    else:
        print(f"CIFAR-10 dataset already loaded with {count} images.")

    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/dataset')
def dataset():
    cursor = get_db().cursor()
    # Modified query to use only columns that exist in the table
    cursor.execute("SELECT filename, label FROM cifar_images LIMIT 100")
    rows = cursor.fetchall()
    headers = [description[0] for description in cursor.description]
    return render_template('dataset.html', rows=rows, headers=headers)

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

@app.route('/references')
def references():
    return render_template('references.html')

if __name__ == '__main__':
    prepare_dataset()
    app.run(debug=True)