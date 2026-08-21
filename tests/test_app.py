import unittest
from unittest.mock import MagicMock, patch

from app import app, db
from research.src.auth import User, ChatSession, Message

class MedicalChatBotTestCase(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SERVER_NAME'] = None  # Prevent host header routing restriction in tests

        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

        # Recreate tables in memory
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_signup_validation_pass(self):
        """Test successful registration with correct inputs."""
        response = self.client.post('/signup', data={
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        # Check that user is created in database
        user = User.query.filter_by(email='test@example.com').first()
        self.assertIsNotNone(user)
        self.assertEqual(user.name, 'test')

    def test_signup_password_mismatch(self):
        """Test signup fails when passwords do not match."""
        response = self.client.post('/signup', data={
            'email': 'mismatch@example.com',
            'password': 'password123',
            'confirm_password': 'password321'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Passwords do not match.", response.data)
        user = User.query.filter_by(email='mismatch@example.com').first()
        self.assertIsNone(user)

    def test_signup_password_too_short(self):
        """Test signup fails when password is under 8 characters."""
        response = self.client.post('/signup', data={
            'email': 'short@example.com',
            'password': 'short',
            'confirm_password': 'short'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Password must be at least 8 characters long.", response.data)
        user = User.query.filter_by(email='short@example.com').first()
        self.assertIsNone(user)

    def test_login_logout(self):
        """Test user login and logout flow."""
        # 1. Create a user
        self.client.post('/signup', data={
            'email': 'user@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        
        # 2. Login
        response = self.client.post('/login', data={
            'email': 'user@example.com',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # 3. Access home (should work since logged in)
        home_resp = self.client.get('/')
        self.assertEqual(home_resp.status_code, 200)
        
        # 4. Logout
        logout_resp = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(logout_resp.status_code, 200)

    @patch('app.mail.send')
    def test_forgot_password_and_reset_flow(self, mock_mail_send):
        """Test OTP generation, mail sending, verification, and password reset."""
        # 1. Create user
        self.client.post('/signup', data={
            'email': 'reset@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        # Log out first to test reset password logged-out flow
        self.client.get('/logout')

        # 2. Trigger forgot password
        forgot_resp = self.client.post('/forgot-password', data={
            'email': 'reset@example.com'
        }, follow_redirects=True)
        
        self.assertEqual(forgot_resp.status_code, 200)
        mock_mail_send.assert_called_once()
        
        # Get generated OTP from database
        user = User.query.filter_by(email='reset@example.com').first()
        self.assertIsNotNone(user.reset_otp)
        otp = user.reset_otp

        # 3. Verify incorrect OTP
        wrong_verify = self.client.post(f'/verify-otp/{user.email}', data={
            'otp': '000000'
        }, follow_redirects=True)
        self.assertIn(b"Invalid OTP.", wrong_verify.data)

        # 4. Verify correct OTP
        correct_verify = self.client.post(f'/verify-otp/{user.email}', data={
            'otp': otp
        }, follow_redirects=True)
        self.assertEqual(correct_verify.status_code, 200)

        # 5. Reset Password
        reset_resp = self.client.post(f'/reset-password/{user.email}', data={
            'password': 'newpassword123',
            'confirm_password': 'newpassword123'
        }, follow_redirects=True)
        self.assertEqual(reset_resp.status_code, 200)
        self.assertIn(b"Password reset successful", reset_resp.data)

        # Verify password updated in DB
        db.session.refresh(user)
        from werkzeug.security import check_password_hash
        self.assertTrue(check_password_hash(user.password_hash, 'newpassword123'))

    @patch('app.classify_intent')
    @patch('app.chatModel')
    def test_chat_session_management(self, mock_chat_model, mock_classify_intent):
        """Test creating, loading, and deleting chat sessions."""
        mock_classify_intent.return_value = 'greeting'
        mock_chat_model.invoke.return_value.content = 'Hello! I am MediAssist.'

        # Create and login user
        self.client.post('/signup', data={
            'email': 'chat@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })

        # Load session should create one if none exists
        with self.client.session_transaction() as sess:
            self.assertNotIn('chat_session_id', sess)

        # Post a message to /get to trigger session creation
        self.client.post('/get', data={'msg': 'Hi'})
        
        # Now there should be an active session
        user = User.query.filter_by(email='chat@example.com').first()
        sessions = ChatSession.query.filter_by(user_id=user.id).all()
        self.assertEqual(len(sessions), 1)
        session_id = sessions[0].id

        # Load session endpoint
        load_resp = self.client.get(f'/load_session/{session_id}')
        self.assertEqual(load_resp.status_code, 200)
        self.assertEqual(load_resp.json['session_id'], session_id)

        # Delete session
        del_resp = self.client.post(f'/delete_session/{session_id}')
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json['success'])

        # Verify deleted in DB
        self.assertIsNone(ChatSession.query.get(session_id))

if __name__ == '__main__':
    unittest.main()
