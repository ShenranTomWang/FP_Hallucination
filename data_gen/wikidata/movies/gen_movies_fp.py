import os, json, random

if __name__ == '__main__':
    load_file = 'dataset/toy_dataset/movies/wikidata_movies.json'
    with open(load_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    unique_years = set()
    for item in data:
        years = item['time']
        years_set = set(years)
        item['time'] = years_set
        unique_years.update(years_set)
    
    for item in data:
        available_choices = unique_years - item['time']
        item['time_fp'] = [random.choice(list(available_choices))]
        item['time'] = list(item['time'])
    
    result_file = 'dataset/toy_dataset/movies/wikidata_movies.json'
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')