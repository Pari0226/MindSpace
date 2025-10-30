from app import db, app

print('Resolved SQLALCHEMY_DATABASE_URI:', app.config.get('SQLALCHEMY_DATABASE_URI'))
print('Resolved DB file path:', app.config.get('DB_FILE_PATH'))

with app.app_context():
    db.create_all()
    print('DB created (or already exists) at:', app.config.get('DB_FILE_PATH'))
