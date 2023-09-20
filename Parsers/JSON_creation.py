import re


data = {}

current_user = None
current_word = None

with open('input_file.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    # Remove leading/trailing whitespace
    line = line.strip()  
    if line.startswith("User:"):
        current_user = line.split()[1].strip()  
        data[current_user] = {}  

        current_word = line.split(":")[1].strip()  
        line_data = line.split()
        current_word = line_data[len(line_data)-1]
        start_match = re.search('W', current_word)
        end_match = re.search('F', current_word)
        extracted_word = current_word[start_match.end(0):end_match.start(0)]
        word_name = 'W'+extracted_word
        if current_user is not None:
            data[current_user][word_name] = {
                "User": current_user,
                "Orientation": line_data[3],  
                "Session": line_data[5],
                "View": line_data[7],
                "FrameRate": line_data[9],
                "FileName": line_data[11],
                "no_of_trials": 0,
                "trials": {} 
            }
    elif line.startswith("trials:"):
        if current_user is not None and word_name is not None:
            trials_count = int(line.split(":")[1].strip())  
            data[current_user][word_name]["no_of_trials"] = trials_count
    elif line.startswith("trial"):
        print("Hola",line)
        line_parts = line.split()
        trail_no = line_parts[0][5:len(line_parts[0])]
        starting = line_parts[1]
        ending = line_parts[2]
        # temp_dict = {trail_no : {'starting' : starting, 'ending' : ending }}
        data[current_user][word_name]["trials"][trail_no] = {'starting' : starting, 'ending' : ending }


import json

json_data = json.dumps(data, indent=4)


with open('output.json', 'w') as json_file:
    json_file.write(json_data)
