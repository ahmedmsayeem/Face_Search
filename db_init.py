import sqlite3

def init_db():
    conn = sqlite3.connect('face_search.db')
    c = conn.cursor()
    
    c.executescript('''
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        link TEXT NOT NULL,
        category_id INTEGER,
        encoding BLOB NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(id)
    );
    ''')
    
    conn.commit()
    conn.close()


init_db()

