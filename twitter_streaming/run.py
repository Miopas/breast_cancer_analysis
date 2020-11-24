import sys
import json
import re
from itertools import permutations

from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from tweepy.streaming import StreamListener


def format_keywords_pattern(arr):
    '''
    Create the regular expression that can match multiple keywords in any order.
    For example, to match the text including 'A', 'B', and 'C', the regular expression would be
    '(A.*B.*C)|(A.*C.*B)|(B.*A.*C)|(B.*C.*A)|(C.*A.*B)|(C.*B.*A)'
    '''
    perm = permutations([i for i in range(len(arr))])
    terms = []
    for order in perm:
        new_arr = []
        for i in order:
            new_arr.append(arr[i])
        terms.append('({})'.format('.*'.join(new_arr)))
    return '({})'.format('|'.join(terms))


filters = [
    format_keywords_pattern(['#?breast', '#?cancer', '#?survivor']),
    format_keywords_pattern(['#?breastcancer', '#?survivor']),
    format_keywords_pattern(['#?tamoxifen', '#?cancer']),
    format_keywords_pattern(['#?tamoxifen', '#?survivor']),
    format_keywords_pattern(['(my|i|me)', '#?breast', '#?cancer']),
    format_keywords_pattern(['(my|i|me)', '#?breastcancer']),
    format_keywords_pattern(['(my|i|me)', '#?tamoxifen'])
]
print(filters)
regex_filter = re.compile(r'|'.join(filters))
rt_filter = re.compile(r'^RT ')


class Listener(StreamListener):
    def __init__(self, output_file=sys.stdout):
        super(Listener,self).__init__()
        self.output_file = output_file

    def on_status(self, status):
        #print(status.text, file=self.output_file)
        if rt_filter.search(status.text) == None and regex_filter.search(status.text.lower()) != None:
            print('{}\t{}\t{}'.format(status.user.id, status.id, status.text), file=self.output_file)

    def on_error(self, status_code):
        print(status_code)
        return False


if __name__ == '__main__':
    # Authentication
    ACCESS_TOKEN = '***'
    ACCESS_TOKEN_SECRET = '***'
    CONSUMER_KEY = '***'
    CONSUMER_SECRET = '***'

    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # Create the streaming listener
    output = open('stream_output.txt', 'w')
    #output = sys.stdout
    listener = Listener(output_file=output)

    stream = Stream(auth=api.auth, listener=listener, tweet_mode='extended')
    try:
        print('Start streaming.')
        #stream.sample(languages=['en'])
        stream.filter(languages=['en'], track=['breast', 'cancer', 'survivor', 'breastcancer', 'tamoxifen'])
    except KeyboardInterrupt as e :
        print("Stopped.")
    finally:
        print('Done.')
        stream.disconnect()
        output.close()

