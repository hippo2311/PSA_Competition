import smtplib
from email.message import EmailMessage
import os

def send_email_to_stakeholder(notify, final=None):
    """
    Send email with optional PDF attachments from final dictionary.

    Args:
        notify: Dictionary with email details
        final: Dictionary containing PDF file paths (optional)
    """

    # Email credentials
    sender_email = "psa.rapidreportbot@gmail.com"
    sender_password = "twcu dgzt yhuz qtap"

    # Create email
    msg = EmailMessage()
    msg['Subject'] = notify.get('subject')
    msg['From'] = sender_email
    msg['To'] = notify.get('to')
    email_body = notify.get("body")

    msg.set_content(email_body)

    # Collect PDF attachments from final dictionary
    if final:
        pdf_paths = [
            final.get("historical_data_ai_solution_pdf"),
            final.get("knowledge_base_ai_solution_pdf"),
            final.get("our_own_ai_solution_pdf")
        ]

        # Filter out None or empty paths
        pdf_paths = [path for path in pdf_paths if path]

        # Attach each PDF
        for file_path in pdf_paths:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    continue

                # Read file and attach
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(file_path)

                    # Add as PDF attachment
                    msg.add_attachment(file_data, 
                                     maintype='application', 
                                     subtype='pdf', 
                                     filename=file_name)
                    print(f"Attached: {file_name}")

            except Exception as e:
                print(f"Error attaching file {file_path}: {e}")

    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent successfully to {notify.get('to')}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False