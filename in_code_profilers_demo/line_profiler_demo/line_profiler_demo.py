import pandas as pd
import line_profiler
import cProfile
import logging

from faker import Faker

fake = Faker()

logging.basicConfig(level=logging.INFO)

df = pd.DataFrame([(fake.name(), fake.phone_number()) for _ in range(1000)], columns=['name', 'phone'])
df.head(3)


def function_with_issues(df):
    logging.debug(f'called with {df}')

    def _get_first_name(n):
        return n.split()[0]

    def _get_last_name(n):
        return ' '.join(n.split()[1:])

    df['first'] = df['name'].apply(_get_first_name)
    df['last'] = df['name'].apply(_get_last_name)

    return df


if __name__ == "__main__":
    print("@@@@ cprofile")
    with cProfile.Profile() as pf:
        function_with_issues(df)
        pf.print_stats()

    print("@@@@ line profile")
    lp = line_profiler.LineProfiler()
    lp.add_function(function_with_issues)


    # this could be any existing function as well, you don't have to write this from scratch
    def wrapper_function():
        function_with_issues(df)


    wrapper = lp(wrapper_function)
    wrapper()
    lp.print_stats()

#demo taken from https://www.wrighters.io/profiling-python-code-with-line_profiler/
