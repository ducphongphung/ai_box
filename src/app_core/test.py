import smtplib


# Set up the SMTP server
smtp_server = "smtp.gmail.com"
port = 465
email = "aiboxmail0@gmail.com"
# password = "aibox2024"

sender_email = email
receiver_email = "ducphongBKEU@gmail.com"

message = """From: aiboxmail0@gmail.com
To: ducphongBKEU@gmail.com
Subject: Fall Email
FBI Warning
"""

# Connect to the SMTP server and send the email
with smtplib.SMTP_SSL(smtp_server, port) as server:
    server.login(email, "agfb qgra wvrb xpwo")
    server.sendmail(sender_email, receiver_email, message)

print("Email sent successfully!")