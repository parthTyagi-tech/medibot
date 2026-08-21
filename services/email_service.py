import os
import urllib.request
import json

def send_async_email(app_instance, msg, mail_extension=None):
    """
    Sends an email asynchronously via Resend, SendGrid, or Flask-Mail SMTP fallback.
    """
    with app_instance.app_context():
        try:
            resend_key = os.getenv("RESEND_API_KEY")
            sendgrid_key = os.getenv("SENDGRID_API_KEY")

            if resend_key:
                app_instance.logger.info("Using Resend HTTP API to send email")
                url = "https://api.resend.com/emails"
                headers = {
                    "Authorization": f"Bearer {resend_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "from": "MediAssist <onboarding@resend.dev>",
                    "to": msg.recipients,
                    "subject": msg.subject,
                    "html": msg.html
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req) as response:
                    resp_data = response.read().decode("utf-8")
                    app_instance.logger.info("Resend HTTP API Success: %s", resp_data)

            elif sendgrid_key:
                app_instance.logger.info("Using SendGrid HTTP API to send email")
                url = "https://api.sendgrid.com/v3/mail/send"
                headers = {
                    "Authorization": f"Bearer {sendgrid_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "personalizations": [{"to": [{"email": r} for r in msg.recipients]}],
                    "from": {"email": "parthtyagi3389@gmail.com"},
                    "subject": msg.subject,
                    "content": [{"type": "text/html", "value": msg.html}]
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req) as response:
                    app_instance.logger.info("SendGrid HTTP API Success")

            else:
                app_instance.logger.info("Falling back to Flask-Mail SMTP")
                if mail_extension:
                    mail_extension.send(msg)
                app_instance.logger.info("SMTP email sent successfully to %s", msg.recipients)

        except Exception as e:
            app_instance.logger.error("OTP email error: %s", e)
