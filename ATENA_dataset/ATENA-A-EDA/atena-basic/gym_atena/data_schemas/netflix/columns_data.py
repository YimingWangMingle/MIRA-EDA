KEYS = ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating',
        'duration', 'listed_in', 'description']
KEYS_ANALYST_STR = KEYS
assert len(KEYS) == len(KEYS_ANALYST_STR)
FILTER_COLS = KEYS
GROUP_COLS = KEYS

NUMERIC_KEYS = {'show_id', 'release_year'}
AGG_KEYS = ['show_id']
AGG_KEYS_ANALYST_STR = ['show_id']
DONT_FILTER_FIELDS = {'show_id'}
