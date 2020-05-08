# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:14:50 2020

@author: Abdelrahman
"""

import pandas as pd



df_confirmed = pd.read_csv("time_series_covid19_confirmed_global.csv")
df_deaths =  pd.read_csv("time_series_covid19_deaths_global.csv")
df_recovered =  pd.read_csv("time_series_covid19_recovered_global.csv")

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True) 
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)


df_confirmed = df_confirmed.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Confirmed")
df_deaths = df_deaths.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Deaths")
df_recovered = df_recovered.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Recovered")

df_confirmed["Deaths"] = df_deaths.Deaths
df_confirmed["Recovered"] = df_recovered.Recovered

df = df_confirmed

confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()

from fbprophet import Prophet

confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])

m = Prophet(interval_width=0.95,yearly_seasonality=True,daily_seasonality=True)
m.fit(confirmed)
future = m.make_future_dataframe(periods=7)

forecast = m.predict(future)

confirmed_forecast_plot = m.plot(forecast)

confirmed_forecast_plot =m.plot_components(forecast)


#Deaths
deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])

m_deaths = Prophet(interval_width=0.95,yearly_seasonality=True,daily_seasonality=True)
m_deaths.fit(deaths)
future_deaths = m_deaths.make_future_dataframe(periods=7)

forecast_deaths = m_deaths.predict(future_deaths)

confirmed_forecast_plot_deaths = m_deaths.plot(forecast_deaths)

confirmed_forecast_plot_deaths =m_deaths.plot_components(forecast_deaths)

#Recovered
recovered.columns = ['ds','y']
recovered['ds'] = pd.to_datetime(recovered['ds'])

m_recovered = Prophet(interval_width=0.95,yearly_seasonality=True,daily_seasonality=True)
m_recovered.fit(recovered)
future_recovered = m_recovered.make_future_dataframe(periods=7)

forecast_recovered = m_recovered.predict(future_recovered)

confirmed_forecast_plot_recovered = m_recovered.plot(forecast_recovered)

confirmed_forecast_plot_recovered =m_recovered.plot_components(forecast_recovered)