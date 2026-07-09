import sqlite3

conn = sqlite3.connect("cifar_data.db")
cur = conn.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS cifar_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        filename TEXT NOT NULL,
        model_used TEXT,
        accuracy REAL
    )
''')


sample_data = [
    ('airplane', 'airplane.png', 'CNN', 89.0),
    ('automobile', 'automobile.png', 'FCNN', 82.5),
    ('cat', 'cat.png', 'CNN', 87.3),
    ('dog', 'dog.png', 'CT-Based', 90.2),
    ('bird', 'bird.png', 'CNN', 88.6),
    ('deer', 'deer.png', 'CT-Based', 85.4)
]

cur.executemany('''
    INSERT INTO cifar_images (label, filename, model_used, accuracy)
    VALUES (?, ?, ?, ?)
''', sample_data)

conn.commit()
conn.close()
print(" Database created and filled")
