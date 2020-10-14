'''
Reference: https://github.com/dmarx/psaw
'''
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd


if __name__ == '__main__':
    api = PushshiftAPI()

    subreddit = 'breastcancer'
    start_epoch=int(dt.datetime(2017, 1, 1).timestamp())

    # Because of the limited sizes of each request, we use an iterative way to get all the
    # submissions.
    gen = api.search_submissions(after=start_epoch,
                                subreddit=subreddit,
                                filter=['created_utc','author', 'selftext'])

    max_response_cache = 10000
    responses = []
    for response in gen:
        try:
            response.selftext
            responses.append(response)
        except:
            continue

        # Set the limitation of maximum responses
        if len(responses) >= max_response_cache:
            break

    output = {'user_id':[], 'timestamp':[], 'text':[]}
    for obj in responses:
        output['user_id'].append(obj.author)
        output['timestamp'].append(obj.created_utc)
        output['text'].append(obj.selftext)

    pd.DataFrame(output).to_csv('data/data_reddit.csv', index=False)
