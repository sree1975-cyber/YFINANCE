#Sector and Industry

import yfinance as yf
import streamlit as st
import pandas as pd

# Title of the app
st.title("Stock Sector and Industry Viewer")

# Dropdown for sectors
sectors = {
    "Technology": "technology",
    "Healthcare": "healthcare",
    "Finance": "finance",
    "Consumer Discretionary": "consumer-discretionary",
    "Consumer Staples": "consumer-staples",
    "Energy": "energy",
    "Utilities": "utilities",
    "Materials": "materials",
    "Industrials": "industrials",
    "Real Estate": "real-estate",
}

# Dropdown for industries
industries = {
    "Software Infrastructure": "software-infrastructure",
    "Software Applications": "software-applications",
    "Semiconductors": "semiconductors",
    "IT Services": "it-services",
    "Telecommunications": "telecommunications",
    "Health Technology": "health-technology",
    "Financial Services": "financial-services",
    "Retail": "retail",
}

# User input for sector and industry
sector_input = st.selectbox("Select Sector:", list(sectors.keys()))
industry_input = st.selectbox("Select Industry:", list(industries.keys()))

# Submit button
if st.button("Submit"):
    try:
        # Fetch sector and industry data
        sector_code = sectors[sector_input]
        industry_code = industries[industry_input]

        sector = yf.Sector(sector_code)
        industry = yf.Industry(industry_code)

        # Display sector information
        st.header(f"{sector_input} Sector Information")
        sector_info = {
            "Key": sector.key,
            "Name": sector.name,
            "Symbol": sector.symbol,
            "Ticker": sector.ticker,
            "Overview": sector.overview,
        }
        st.table(pd.DataFrame.from_dict(sector_info, orient='index', columns=["Value"]))
        
        # Display top companies in a separate table
        st.subheader("Top Companies")
        st.table(sector.top_companies)

        # Display top ETFs in a separate table
        st.subheader("Top ETFs")
        st.table(sector.top_etfs)

        # Display top mutual funds in a separate table
        st.subheader("Top Mutual Funds")
        st.table(sector.top_mutual_funds)

        # Display industries in a separate table
        st.subheader("Industries")
        st.table(sector.industries)

        # Display industry information
        st.header(f"{industry_input} Industry Information")
        industry_info = {
            "Sector Key": industry.sector_key,
            "Sector Name": industry.sector_name,
        }
        st.table(pd.DataFrame.from_dict(industry_info, orient='index', columns=["Value"]))

        # Display top performing companies in a separate table
        st.subheader("Top Performing Companies")
        st.table(industry.top_performing_companies)

        # Display top growth companies in a separate table
        st.subheader("Top Growth Companies")
        st.table(industry.top_growth_companies)

    except Exception as e:
        st.error(f"An error occurred: {e}")
