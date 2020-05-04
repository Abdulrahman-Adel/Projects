# -*- coding: utf-8 -*-

weather_data_tags_dict = {
    'observation_time': '',
    'weather': '',
    'temp_f': '',
    'temp_c': '',
    'dewpoint_f': '',
    'dewpoint_c': '',
    'relative_humidity': '',
    'wind_string': '',
    'visibility_mi': '',
    'pressure_string': '',
    'pressure_in': '',
    'location': ''
    }

import urllib.request
from datetime import datetime

def get_weather_data(station_id='KSEA'):
    url_general = 'http://w1.weather.gov/xml/current_obs/{}.xml' 
    url = url_general.format(station_id)
    print(url)
    request = urllib.request.urlopen( url )
    content = request.read().decode()   
    
    import xml.etree.ElementTree as ET
    xml_root = ET.fromstring(content)
    
    for data_point in weather_data_tags_dict.keys():
        weather_data_tags_dict[data_point] = xml_root.find(data_point).text 
        
    icon_url_base = xml_root.find("icon_url_base").text    
    icon_url_name = xml_root.find("icon_url_name").text    
    icon_url = icon_url_base + icon_url_name
    
    return weather_data_tags_dict,icon_url

def create_html_report(data_dict,icon_url,html_file):
    
    alt_var = data_dict["weather"]
    
    with open(html_file,mode="w") as outfile:
        outfile.write('\t<tr><td align="center">' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</td></tr></br>\n")
        outfile.write("<img alt={} src={}>".format(alt_var,icon_url))
        outfile.write('<br><span style="color:blue"><b>\tWeather Data:</b>\n')
        outfile.write("<br>")
        outfile.write("<html><table border=1>\n")
        
        for key,value in data_dict.items():
            outfile.write('<tr><td><b><span style="color:black">{:s}</b></td><td align="left"><span style="color:blue"><b>{:s}</b></td></tr>\n'.format(key,value))
        outfile.write("</table></html>\n")    
        
import smtplib 
from email.mime.text import MIMEText
#from GMAIL_PWD import GMAIL_PWD

def send_gmail(msg_file):
    with open(msg_file, mode="rb") as message:
        msg = MIMEText(message.read(),"html","html")
    
    msg["Subject"] = "Hourly Weather: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))    
    msg["From"] = "ONE6@gmail.com"
    msg["To"] = "ANOTHERONE@gmail.com" #DJKHALED :D
    
    server = smtplib.SMTP("smtp.gmail.com",port=587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login("ONE@gmail.com","PASSWORD")
    server.send_message(msg)
    server.close() 
    
#NOTE: you have to Allow access to "less secure apps" in both to_gmail and from_gmail accounts in order for this to work

        
if __name__ == "__main__":
    weather_dict,icon = get_weather_data()
    email_file = "Test_Email_File.html"
    create_html_report(weather_dict,icon,email_file)
    #send_gmail(email_file)
    
    from collections import OrderedDict
    from time import sleep
    from pprint import pprint
    import schedule

    def job():
        pprint(schedule.jobs)
        weather_dict_ordered = OrderedDict(sorted(weather_dict.items()))
        send_gmail(email_file)

    schedule.every(1).minutes.do(job)   

    while True:
        schedule.run_pending()
        sleep(1)     
