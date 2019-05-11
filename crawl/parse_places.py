import json
import re

class abbrevation_to_long:
    
    def __init__(self):
        long_abbrev = {
            'Alabama': 'AL',
            'Alaska': 'AK',
            'Arizona': 'AZ',
            'Arkansas': 'AR',
            'California': 'CA',
            'Colorado': 'CO',
            'Connecticut': 'CT',
            'Delaware': 'DE',
            'Florida': 'FL',
            'Georgia': 'GA',
            'Hawaii': 'HI',
            'Idaho': 'ID',
            'Illinois': 'IL',
            'Indiana': 'IN',
            'Iowa': 'IA',
            'Kansas': 'KS',
            'Kentucky': 'KY',
            'Louisiana': 'LA',
            'Maine': 'ME',
            'Maryland': 'MD',
            'Massachusetts': 'MA',
            'Michigan': 'MI',
            'Minnesota': 'MN',
            'Mississippi': 'MS',
            'Missouri': 'MO',
            'Montana': 'MT',
            'Nebraska': 'NE',
            'Nevada': 'NV',
            'New Hampshire': 'NH',
            'New Jersey': 'NJ',
            'New Mexico': 'NM',
            'New York': 'NY',
            'North Carolina': 'NC',
            'North Dakota': 'ND',
            'Ohio': 'OH',
            'Oklahoma': 'OK',
            'Oregon': 'OR',
            'Pennsylvania': 'PA',
            'Rhode Island': 'RI',
            'South Carolina': 'SC',
            'South Dakota': 'SD',
            'Tennessee': 'TN',
            'Texas': 'TX',
            'Utah': 'UT',
            'Vermont': 'VT',
            'Virginia': 'VA',
            'Washington': 'WA',
            'West Virginia': 'WV',
            'Wisconsin': 'WI',
            'Wyoming': 'WY',
            'Mexico': 'MX'
        }
        self.abbrev_long = {}
        for k, v in long_abbrev.items():
            self.abbrev_long[v] = k
    
    def get_long_name(self, short_name):
        if len(short_name) <= 2:
            if self.abbrev_long.has_key(short_name):
                return self.abbrev_long[short_name]
            else:
                return short_name
        return short_name
            
            

def read_earthquake_json(json_file, output_file):
    find_city_pattern = re.compile(r'.* of (.*)')

    # 1. First load from json
    data = json.load(open(json_file))
    features = data['features']
    earthquake_list = []
    iteration = 0
    state_or_country = {}
    abb_to_long = abbrevation_to_long()
    for feature in features:
        properties = feature['properties']

        # 1.2 Just fetch from Twitter when the place matches expression ".* of .*"
        # Usually places have names like "200km of Oakton, Virginia" 
        matches = find_city_pattern.match(properties['place'])
        if matches:
            actual_city = matches.group(1)
            elem = actual_city.split(', ')
            q_place = abb_to_long.get_long_name(elem[-1])
            if not state_or_country.has_key(q_place):
                state_or_country[q_place] = True
    
    for k,v in state_or_country.items():         
        print(k)


file_name = "earthquake_world_2019_mag5_count506.json"
file_without_ext = file_name.split(".")

output_file = "./Tweets/Tweets_" + file_without_ext[0] + ".json"
earthquake_data_json_path = "./UsgsData/" +file_name
