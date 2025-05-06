class Config:
    SECRET_KEY = 'dev'  # Replace this with a secure value in production

    # MySQL Database configuration (AWS RDS)
    DB_USER = 'admin'
    DB_PASSWORD = 'databasepassword_g21'
    DB_HOST = 'database-tirp.c1gieoo4asys.us-east-1.rds.amazonaws.com'
    DB_PORT = '3306'
    DB_NAME = 'tirp'

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False