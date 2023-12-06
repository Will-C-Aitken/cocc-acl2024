import json
import re
import os

def main():

    survey_file = os.path.join("data", "emnlp2022_survey.json")
    with open(survey_file, 'r') as f:
        contents = f.read()

    parsed_contents = json.loads(contents)

    for p in parsed_contents["papers"]:
        
        if "datasets" not in p:
            continue

        for d in p["datasets"]:

            d_link = d["link"]
            year = get_year(d_link)
            d["year"] = year

            
            if "metrics" not in d:
                continue

            for m in d["metrics"]:

                if "prior_best" not in m:
                    continue
                
                pb = m["prior_best"]
                if type(pb) is list:
                    for pb_element in pb:
                        pb_link = pb_element["link"]
                        year = get_year(pb_link)
                        pb_element["year"] = year
                else:
                    pb_link = pb["link"]
                    year = get_year(pb_link)
                    pb["year"] = year

            for m in d["metrics"]:

                if "full_setting" not in m:
                    continue
                
                fs = m["full_setting"]
                if type(fs) is list:
                    for fs_element in fs:
                        fs_link = fs_element["link"]
                        year = get_year(fs_link)
                        fs_element["year"] = year
                else:
                    fs_link = fs["link"]
                    year = get_year(fs_link)
                    fs["year"] = year

        if "basemodel" not in p:
            continue

        for b in p["basemodel"]:

            b_link = b["link"]
            year = get_year(b_link)
            b["year"] = year


    with open('survey_w_years.json', 'w') as fp:
        json.dump(parsed_contents, fp, indent=4)

def get_year(link):
    
    #arxiv
    match = re.search(r"arxiv.*/pdf/([0-9]{2})[0-9]{2}\.", link)
    if match is not None:
        return "20" + match.group(1)

    # acl XYR format where X is location first letter and YR is year
    match = re.search(r"aclanthology.*[A-Z]([0-9]{2})-", link)
    if match is not None:
        return "20" + match.group(1)

    # acl *org/YEAR. format where YEAR is year
    match = re.search(r"aclanthology\.org/([0-9]{4})\.", link)
    if match is not None:
        return match.group(1)

if __name__ == "__main__":
    main()
