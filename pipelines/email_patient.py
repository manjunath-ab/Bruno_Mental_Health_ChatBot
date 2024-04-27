import smtplib
import ssl
from email.message import EmailMessage
 
# Define email sender and receiver
email_sender = 'zenaidemo111@gmail.com'
email_password = 'bjzl qsvy bkux ojvz'

 
# Set the subject and body of the email
subject = 'ZenAI'
body = """
Hi I'm Zenny! Here are your appointment details:
"""
def def_email_message(email_receiver,date_time,therapist_name):
 em = EmailMessage()
 em['From'] = email_sender
 em['To'] = email_receiver
 em['Subject'] = "Therapist Appointment Scheduled"
 body=f"This is to inform you that your appointment has been scheduled on {date_time[0]} at {date_time[1]} with {therapist_name}."
 em.set_content(body)
 return em
# Add SSL (layer of security)
context = ssl.create_default_context()
 
# Log in and send the email
def send_email(email_receiver,date_time,therapist_name):
 
 em=def_email_message(email_receiver,date_time,therapist_name)
 with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
    smtp.login(email_sender, email_password)
    smtp.sendmail(email_sender, email_receiver, em.as_string())