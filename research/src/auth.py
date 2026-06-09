import os
from flask_dance.contrib.google import make_google_blueprint
from flask_login import LoginManager, UserMixin
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()
login_manager = LoginManager()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    email = db.Column(
        db.String(250),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=True
    )
    reset_otp = db.Column(
        db.String(10),
        nullable=True
    )

    otp_expiry = db.Column(
        db.DateTime,
        nullable=True
    )

    name = db.Column(db.String(250))

    avatar = db.Column(
        db.String(500),
        default=""
    )

    memory = db.Column(
        db.Text,
        default=""
    )

    sessions = db.relationship(
        'ChatSession',
        backref='user',
        lazy=True
    )
    memory = db.Column(db.Text, default="")
    sessions = db.relationship('ChatSession', backref='user', lazy=True)
    
class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default="New Consultation")
    summary = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='session', lazy=True, cascade="all, delete-orphan")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(20))   # 'user' or 'assistant'
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_google_blueprint():
    return make_google_blueprint(
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scope=["profile", "email"],
        redirect_to="google_auth"
    )