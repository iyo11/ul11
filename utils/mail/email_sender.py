import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from pathlib import Path
import yaml


def load_config():
    # Get the absolute path of the current script
    current_dir = Path(__file__).resolve().parent

    # email.yaml is located at ../../config/email.yaml
    config_path = current_dir.parent.parent / "config/email.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.absolute()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# Load configuration
try:
    config = load_config()
    SMTP_SERVER = config["email"]["smtp_server"]
    SMTP_PORT = config["email"]["smtp_port"]
    SENDER_EMAIL = config["email"]["sender_email"]
    PASSWORD = config["email"]["password"]
except Exception as e:
    print(f"Failed to initialize email config: {e}")
    exit(1)


def send_text_email(receiver, subject, content):
    message = MIMEText(content, "plain", "utf-8")
    if isinstance(receiver, (list, tuple)):
        receivers = list(receiver)
        to_header = ", ".join(receivers)
    else:
        receivers = [receiver]
        to_header = receiver
    # RFC-compliant From header: display name + email address
    message["From"] = formataddr(
        (str(Header("YOLO Notification", "utf-8")), SENDER_EMAIL)
    )
    message["To"] = to_header
    message["Subject"] = Header(subject, "utf-8")

    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, [receiver], message.as_string())
        server.quit()

        print(f"Email successfully sent to: {receiver}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("Email send failed: authentication error. Check email address or app password.")
        return False

    except Exception as e:
        print(f"Email send failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing email module...")
    send_text_email(
        "iyo11@qq.com",
        "Test Email",
        "This is a self-test email from the mail module."
    )
