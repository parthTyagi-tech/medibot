def register_routes(app):
    from routes.auth import auth_bp
    from routes.chat import chat_bp
    from routes.voice import voice_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(voice_bp)

    # Alias endpoints for backward compatibility with root-level url_for('login'), url_for('index'), etc.
    app.add_url_rule('/', endpoint='index', build_only=True)
    app.add_url_rule('/login', endpoint='login', build_only=True)
    app.add_url_rule('/signup', endpoint='signup', build_only=True)
    app.add_url_rule('/forgot-password', endpoint='forgot_password', build_only=True)
    app.add_url_rule('/verify-otp/<email>', endpoint='verify_otp', build_only=True)
    app.add_url_rule('/reset-password/<email>', endpoint='reset_password', build_only=True)
    app.add_url_rule('/logout', endpoint='logout', build_only=True)
    app.add_url_rule('/google_auth', endpoint='google_auth', build_only=True)
