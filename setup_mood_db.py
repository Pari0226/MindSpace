import sqlite3

def create_mood_table():
    conn = sqlite3.connect('instance/mood.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS moods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            date DATE NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_mood_table()
    print("moods table created successfully.")
