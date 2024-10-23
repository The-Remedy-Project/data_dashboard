import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import json

from requests_html import HTMLSession

import logging


# Define a function to handle the title casing and replacements
def format_facility_name(name):
    # Define a dictionary of words to replace
    replacements = {
        'CTR': 'CENTER',
        'CORR': 'CORRECTIONAL',
        'FCL': 'FACILITY',
        'FACL': 'FACILITY',
        'DIST': 'DISTRICT',
    }

    # Replace specific abbreviations with full words
    for abbr, full in replacements.items():
        name = re.sub(rf'\b{abbr}\b', full, name, flags=re.IGNORECASE)
    
    # First, convert the string to title case
    name = name.title()

    # Define a list of words to keep in all caps
    exception_list = ['CCM', 'FCI', 'CI', 'USP', 'FMC', 'FPC', 'DC', 'MDC', 'FDC', 'FTC', 'MCC', 'FL', 'II', 'III', 'ADMAX', 'USMCFP', 'NE', 'RO', 'HQ']
    exception_list += ['of', 'and', 'the', 'for', 'in', 'at', 'by']
    exception_list += ['McRae', 'McCreary', 'McDowell', 'McKean']
    # Use regex to preserve words that should remain in all caps
    for word in exception_list:
        name = re.sub(rf'\b{word.title()}\b', word, name)

    # keep_all_lower = ['of', 'and', 'the', 'for', 'in', 'at', 'by']

    # for word in keep_all_lower:
    #     name = re.sub(rf'\b{word.title()}\b', word, name)
    
    return name

# Function to adjust lat and long for duplicates
def adjust_lat_long(df, radius):
    # Group by lat and long to find duplicates
    duplicates = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')
    duplicates = duplicates[duplicates['count'] > 1]
    
    # Create new columns for adjusted latitudes and longitudes
    df['lat_adj'] = df['latitude']
    df['long_adj'] = df['longitude']

    for _, row in duplicates.iterrows():
        # Get the rows with the same lat and long
        same_location = df[(df['latitude'] == row['latitude']) & (df['longitude'] == row['longitude'])]

        # Number of points at this location
        num_points = len(same_location)

        # Calculate angle increment
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # Adjust lat and long for each duplicate
        for i, (index, _) in enumerate(same_location.iterrows()):
            df.loc[index, 'lat_adj'] = df.loc[index, 'latitude'] + (radius * np.sin(angles[i]))
            df.loc[index, 'long_adj'] = df.loc[index, 'longitude'] + (radius * np.cos(angles[i]))

    return df

# get rid of DEBUG output from the requests_html queries
logger_pyp = logging.getLogger("pyppeteer")
logger_pyp.setLevel(logging.WARNING)
logger_wbs = logging.getLogger("websockets")
logger_wbs.setLevel(logging.WARNING)

regional_office_codes = ['MXR', 'NCR', 'NER', 'SCR', 'SER', 'WXR']
central_office_code = 'BOP'

name_key_df = pd.read_csv('../data/facility-codes-updated.csv',)

name_key_df['nice_name'] = name_key_df['facility_name'].apply(format_facility_name)

session = HTMLSession()
resp = session.get('https://www.bop.gov/locations/map.jsp')
resp.html.render(sleep = 10)
data = resp.html.find('script')[6]
data = re.search(r'var allGPS = (\[.*\])', data.text).group(1)

# 1. Replace `//complex:""` comments with empty space
data_cleaned = re.sub(r', //complex:"[^"]*"\}', '', data)

# 2. Replace invalid dictionary keys and values with proper double quotes
data_cleaned = re.sub(r'(\w+):', r'"\1":', data_cleaned)

# 3. Remove empty strings from lists
data_cleaned = data_cleaned.replace(', "", ""', '')

# Remove redundant dims
data_cleaned = data_cleaned.replace('[{', '{')
data_cleaned = data_cleaned.replace('}]', '}')

# 4. Convert to a valid Python list
try:
    data_list = eval(data_cleaned)  # Using eval for simplicity, ensure the data is truste
except Exception as e:
    print(f"Error parsing data: {e}")

inst_loc_dict = {}
for item in data_list:
    inst_loc_dict[item['code']] = item
    inst_loc_dict[item['code']].pop('code', None)

# Extract options as a dictionary with value:text pairs
inst_code_dict = {inst_code.attrs['value']: inst_code.text for inst_code in resp.html.find('select#select_facil', first=True).find('option')[1:]}

name_key_df['latitude'] = [float(inst_loc_dict.get(code).get('latitude')) if inst_loc_dict.get(code,None) is not None else None for code in name_key_df['facility_code']]
name_key_df['longitude'] = [float(inst_loc_dict.get(code).get('longitude')) if inst_loc_dict.get(code,None) is not None else None for code in name_key_df['facility_code']]
name_key_df['type'] = [inst_loc_dict.get(code).get('type') if inst_loc_dict.get(code,None) is not None else None for code in name_key_df['facility_code']]
name_key_df['securityType'] = [inst_loc_dict.get(code).get('securityType') if inst_loc_dict.get(code,None) is not None else None for code in name_key_df['facility_code']]

for new_col_name in ['addr1', 'addr2', 'city', 'state', 'zip', 'pop_total']:
    name_key_df[new_col_name] = pd.Series()

base_url = 'https://www.bop.gov/locations'
for i, fac_code in name_key_df.loc[:, 'facility_code'].items(): #[73:74]:
    print(fac_code)
    print(f'{i+1}/{len(name_key_df)}')
    if fac_code in regional_office_codes:
        r = session.get(f'{base_url}/regional_offices/{fac_code.lower()}o/')
    elif fac_code == central_office_code:
        r = session.get(f'{base_url}/central_office/')
    else:
        r = session.get(f'{base_url}/{fac_code.lower()}/')
        if int(r.status_code) != 200:
            r = session.get(f'{base_url}/ccm/{fac_code.lower()}/')
            if int(r.status_code) != 200:
                r = session.get(f'{base_url}/institutions/{fac_code.lower()}/')
                if int(r.status_code) != 200:
                    print('failed failed failed')
                    continue
    if r.url == 'https://www.bop.gov/locations/index.jsp':
        continue
    r.html.render(sleep = 10) # sleeping is optional but do it just in case
    name_key_df.at[i, 'addr1'] = r.html.find('div#address', first=True).text
    if r.html.find('div#address2', first=True):
        name_key_df.at[i, 'addr2'] = r.html.find('div#address2', first=True).text
    name_key_df.at[i, 'city'] = r.html.find('span#city', first=True).text
    name_key_df.at[i, 'state'] = r.html.find('span#state', first=True).text
    name_key_df.at[i, 'zip'] = r.html.find('span#zip_code', first=True).text
    name_key_df.at[i, 'latitude'] = float(r.html.find('span#latitude', first=True).text)
    name_key_df.at[i, 'longitude'] = float(r.html.find('span#longitude', first=True).text)

    
    pop_total = r.html.find('td#pop_count.pop-label.pop-total', first=True)
    if pop_total is None:
        pop_total = r.html.find('td#pop_count.pop-label', first=True)
    if pop_total is not None:
        name_key_df.at[i, 'pop_total'] = int(pop_total.text.replace(',',''))

name_key_df[['pop_total']] = name_key_df[['pop_total']].apply(pd.to_numeric, downcast='integer')

# Radius for adjustment (in degrees, a small value)
radius = 0.002

# Apply the function
name_key_df = adjust_lat_long(name_key_df, radius)

name_key_df.to_csv('../data/facility-info.csv')