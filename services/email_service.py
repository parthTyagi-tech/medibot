import os
import urllib.request
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def send_async_email(app_instance, msg, mail_extension=None):
    """
    Sends an email asynchronously via Resend, SendGrid, or Flask-Mail SMTP fallback.
    """
    with app_instance.app_context():
        try:
            brevo_key = os.getenv("BREVO_API_KEY") or os.getenv("SENDINBLUE_API_KEY")
            resend_key = os.getenv("RESEND_API_KEY")
            sendgrid_key = os.getenv("SENDGRID_API_KEY")
            sender_email = os.getenv("MAIL_USERNAME") or os.getenv("MAIL_DEFAULT_SENDER") or "parthtyagi3389@gmail.com"

            if brevo_key:
                app_instance.logger.info("Using Brevo (Sendinblue) HTTP API to send OTP email")
                url = "https://api.brevo.com/v3/smtp/email"
                headers = {
                    "api-key": brevo_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                payload = {
                    "sender": {"name": "MediAssist AI", "email": sender_email},
                    "to": [{"email": r} for r in msg.recipients],
                    "subject": msg.subject,
                    "htmlContent": msg.html
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req) as response:
                    resp_data = response.read().decode("utf-8")
                    app_instance.logger.info("Brevo HTTP API Success: %s", resp_data)

            elif resend_key:
                app_instance.logger.info("Using Resend HTTP API to send email")
                url = "https://api.resend.com/emails"
                headers = {
                    "Authorization": f"Bearer {resend_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "from": f"MediAssist <{sender_email}>",
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
                    "from": {"email": sender_email},
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


def send_eval_alert(subject: str, html_content: str, recipient_email: Optional[str] = None, app_instance=None) -> bool:
    """
    Sends an immediate critical evaluation safety alert email.
    Reuses the existing Brevo -> Resend -> SendGrid -> Flask-Mail fallback hierarchy.
    """
    brevo_key = os.getenv("BREVO_API_KEY") or os.getenv("SENDINBLUE_API_KEY")
    resend_key = os.getenv("RESEND_API_KEY")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    sender_email = os.getenv("MAIL_USERNAME") or os.getenv("MAIL_DEFAULT_SENDER") or "parthtyagi3389@gmail.com"
    target_email = recipient_email or os.getenv("EVAL_ALERT_EMAIL") or sender_email

    log = app_instance.logger if app_instance and hasattr(app_instance, "logger") else logger

    try:
        if brevo_key:
            url = "https://api.brevo.com/v3/smtp/email"
            headers = {
                "api-key": brevo_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "sender": {"name": "MediAssist Clinical Safety Monitor", "email": sender_email},
                "to": [{"email": target_email}],
                "subject": f"[CLINICAL SAFETY ALERT] {subject}",
                "htmlContent": html_content
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                resp_data = response.read().decode("utf-8")
                log.info("Brevo alert email sent: %s", resp_data)
                return True

        elif resend_key:
            url = "https://api.resend.com/emails"
            headers = {
                "Authorization": f"Bearer {resend_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "from": f"MediAssist Safety <{sender_email}>",
                "to": [target_email],
                "subject": f"[CLINICAL SAFETY ALERT] {subject}",
                "html": html_content
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                log.info("Resend alert email sent successfully")
                return True

        elif sendgrid_key:
            url = "https://api.sendgrid.com/v3/mail/send"
            headers = {
                "Authorization": f"Bearer {sendgrid_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "personalizations": [{"to": [{"email": target_email}]}],
                "from": {"email": sender_email},
                "subject": f"[CLINICAL SAFETY ALERT] {subject}",
                "content": [{"type": "text/html", "value": html_content}]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                log.info("SendGrid alert email sent successfully")
                return True
        else:
            log.warning("No email provider API keys configured for send_eval_alert.")
            return False

    except Exception as e:
        log.error("Failed to send eval alert email: %s", e)
        return False

