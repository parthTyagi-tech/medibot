import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Set test environment flags and secrets BEFORE importing application modules
os.environ["TESTING"] = "1"
if not os.getenv("SECRET_KEY"):
    os.environ["SECRET_KEY"] = "test-secret-key-for-automated-suites-32chars"
if not os.getenv("VOICE_INTERNAL_SECRET"):
    os.environ["VOICE_INTERNAL_SECRET"] = "test-voice-secret-key-for-test-suites-32chars"

import pytest

@pytest.fixture
def app_instance():
    from app import app as flask_app, db
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["SERVER_NAME"] = None

    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def test_client(app_instance):
    return app_instance.test_client()
